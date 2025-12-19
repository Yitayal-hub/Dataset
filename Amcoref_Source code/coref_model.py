from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_addons as tfa
import tensorflow_hub as hub
import h5py

import util
import conll
import metrics

class CorefModel(object):
    def __init__(self, config):
        self.config = config
        metrics.INCLUDE_SINGLETONS = self.config['include_singletons']

        global coref_ops
        if config['use_gold_mentions']:
            print(' /!\\ using gold mentions /!\\ ')
            import coref_ops_gold_mentions as coref_ops
        else:
            print(' /!\\ **not** using gold mentions /!\\ ')
            import coref_ops

        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
                
        # NER configuration
        self.ner_types = config["ner_types"]  # e.g. ["PER", "ORG", "LOC", "MISC"]
        self.ner_dict = {ner: i+1 for i, ner in enumerate(self.ner_types)}  # 0 reserved for "no entity"
        self.ner_embedding_size = config["ner_embedding_size"]
        self.ner_lstm_size = config["ner_lstm_size"]#, 256)
        self.ner_use_crf = config["ner_use_crf"]#, True)
        
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.eval_data = None

        input_props = []
        input_props.append((tf.string, [None, None]))  # tokens
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # context_word_emb
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # head_word_emb
        input_props.append((tf.int32, [None, None, None]))  # char_index
        input_props.append((tf.int32, [None]))  # text_len
        input_props.append((tf.int32, [None]))  # speaker_ids
        input_props.append((tf.int32, []))  # genre
        input_props.append((tf.bool, []))  # is_training
        input_props.append((tf.int32, [None]))  # gold_starts
        input_props.append((tf.int32, [None]))  # gold_ends
        input_props.append((tf.int32, [None]))  # cluster_ids
        input_props.append((tf.int32, [None, None]))  # ner_indices (optional)
        
        self.queue_input_tensors = [tf.compat.v1.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.queue.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.compat.v1.assign(self.global_step, 0)
        self.learning_rate = tf.compat.v1.train.exponential_decay(
            self.config["learning_rate"], 
            self.global_step,
            self.config["decay_frequency"], 
            self.config["decay_rate"],
            staircase=True
        )
        
        # Training ops
        self.trainable_variables = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=self.loss, xs=self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.compat.v1.train.AdamOptimizer,
            "sgd": tf.compat.v1.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](self.learning_rate)
        opt_op = optimizer.apply_gradients(zip(gradients, self.trainable_variables), 
                                        global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=self.config["ema_decay"])
        with tf.control_dependencies([opt_op]):
            self.train_op = self.ema.apply(self.trainable_variables)
        self.gold_loss = tf.constant(0.)

    def build_ner_module(self, context_outputs, text_len_mask, ner_indices, is_training):
        """
        Builds the NER module of the model.

        :param context_outputs: Contextualized word embeddings from the LSTM.
        :param text_len_mask: Mask of word lengths.
        :param ner_indices: The ground truth NER labels (already flattened).
        :param is_training: Whether the model is in training mode.
        :return: NER logits, labels, and predictions.
        """
        with tf.compat.v1.variable_scope("ner_module"):
            num_sentences = tf.shape(input=text_len_mask)[0]
            max_sentence_length = tf.shape(input=text_len_mask)[1]

            dropout = self.get_dropout(self.config["dropout_rate"], is_training)

            # Compute sequence_lengths from the flattened context_outputs shape.
            sequence_lengths = tf.cast(tf.shape(context_outputs)[0], tf.int32)
            
            with tf.compat.v1.variable_scope("fw_cell"):
                # Fix: Change batch_size from num_sentences to 1 to match the expanded input
                lstm_cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], 1, self.lstm_dropout)
            with tf.compat.v1.variable_scope("bw_cell"):
                # Fix: Change batch_size from num_sentences to 1 to match the expanded input
                lstm_cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], 1, self.lstm_dropout)
            
            # The input to the RNN is now [1, num_words, embedding_size]
            # so the sequence_length must also be [1].
            (ner_fw, ner_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                tf.expand_dims(context_outputs, 0),
                sequence_length=tf.expand_dims(sequence_lengths, 0),
                dtype=tf.float32)

            ner_outputs = tf.concat([tf.squeeze(ner_fw, 0), tf.squeeze(ner_bw, 0)], axis=1)
            ner_outputs = tf.nn.dropout(ner_outputs, dropout)

            crf_logits = ner_outputs

            # The ner_indices input is already flattened, so we don't need to call flatten_emb_by_sentence again.
            crf_labels = ner_indices

            # Pass the correctly shaped sequence_lengths tensor to the CRF.
            log_likelihood, transition_params = tfa.text.crf_log_likelihood(
                inputs=tf.expand_dims(crf_logits, 0),
                tag_indices=tf.expand_dims(crf_labels, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0)
            )

            viterbi_sequence, _ = tfa.text.crf_decode(
                tf.expand_dims(crf_logits, 0), transition_params, tf.expand_dims(sequence_lengths, 0)
            )

            ner_loss = tf.reduce_mean(input_tensor=-log_likelihood)

            return tf.squeeze(viterbi_sequence, 0), ner_loss

    def multi_head_attention(self, queries, keys, values, num_heads, attention_size, 
                           dropout_rate, scope="multi_head_attention"):
        with tf.compat.v1.variable_scope(scope):
            initial_queries_shape = tf.shape(queries)
            
            if queries.shape[-1] != attention_size:
                query_projection = tf.keras.layers.Dense(attention_size, activation=None, name="query_projection")
                key_projection = tf.keras.layers.Dense(attention_size, activation=None, name="key_projection")
                value_projection = tf.keras.layers.Dense(attention_size, activation=None, name="value_projection")

                queries_proj = query_projection(queries)
                keys_proj = key_projection(keys)
                values_proj = value_projection(values)
            else:
                queries_proj = queries
                keys_proj = keys
                values_proj = values
            
            Q = tf.keras.layers.Dense(attention_size, activation=tf.nn.relu)(queries_proj)
            K = tf.keras.layers.Dense(attention_size, activation=tf.nn.relu)(keys_proj)
            V = tf.keras.layers.Dense(attention_size, activation=tf.nn.relu)(values_proj)
            
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, initial_queries_shape[1], 1])
            
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, initial_queries_shape[1]])
            outputs *= query_masks
            
            outputs = tf.nn.dropout(outputs, rate=1 - (dropout_rate))
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            outputs += queries_proj
            outputs = tf.keras.layers.LayerNormalization(axis=-1)(outputs)

        return outputs

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_example(example, is_training=True)
                    if tensorized_example is None:
                        continue
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()
        self.enqueue_thread = enqueue_thread

    def close(self):
        if hasattr(self, 'enqueue_thread'):
            self.enqueue_thread.join(timeout=1)

    def restore(self, session, latest_checkpoint=False, fpath=None):
        if fpath:
            checkpoint_path = fpath
        else:
            if latest_checkpoint:
                checkpoint_path = tf.train.latest_checkpoint(self.config["log_dir"])
            else:
                checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.compat.v1.global_variables_initializer())

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]

        if self.config['use_gold_mentions']:
            if self.config['include_singletons']:
                min_length = 1
            else:
                min_length = 2

            new_clusters = []
            for cluster in clusters:
                new_cluster = []
                for start, end in cluster:
                    if end - start < self.config["max_span_width"]:
                        new_cluster.append((start, end))
                if len(new_cluster) >= min_length:
                    new_clusters.append(new_cluster)
            clusters = new_clusters
            if not len(clusters):
                return None
            example["clusters"] = clusters

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_sentences = len(sentences)
        max_sentence_length = max(len(s) for s in sentences)
        subword_sentences = []
        for sentence in sentences:
             subword_sentence = [util.get_subword_indices(word, self.subword_vocab_path) for word in sentence]
             subword_sentences.append(subword_sentence)
        num_words = sum(len(s) for s in sentences)

        if num_words * self.config['top_span_ratio'] < 1:
            return None

        speakers = util.flatten(example["speakers"])
        assert num_words == len(speakers)

        # Process NER tags (if available)
        if "ner" in example:
            ner_tags = example["ner"]
            flattened_ner_tags = util.flatten(ner_tags)
            ner_indices = np.array([self.ner_dict.get(tag, 0) for tag in flattened_ner_tags])  # 0 = no entity
        else:
            # Create dummy NER tags if not provided
            ner_tags = [["O"] * len(sent) for sent in example["sentences"]]
            ner_indices = np.zeros(sum(len(s) for s in example["sentences"]), dtype=np.int32)

        # Pad NER indices to match sentence structure
        ner_indices_padded = np.full((num_sentences, max_sentence_length), -1)
        for i, sent in enumerate(ner_tags):
            ner_indices_padded[i, :len(sent)] = [self.ner_dict.get(tag, 0) for tag in sent]

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), 1)
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        max_subwords = max(max(len(subwords) for subwords in util.flatten(subword_sentences)), 1)
        subword_index = np.zeros([len(sentences), max_sentence_length, max_subwords], dtype=np.int32)
        
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
                subword_indices = subword_sentences[i][j]
                subword_index[i, j, :len(subword_indices)] = subword_indices
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        example_tensors = (
            tokens, context_word_emb, head_word_emb, char_index, subword_index, text_len, speaker_ids, genre, is_training,
            gold_starts, gold_ends, cluster_ids, ner_indices_padded)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            res = self.truncate_example(*example_tensors)
            if res is None and self.config['use_gold_mentions']:
                return None
            return res
        else:
            return example_tensors

    def truncate_example(self, tokens, context_word_emb, head_word_emb, char_index, subword_index, text_len, speaker_ids,
                         genre, is_training, gold_starts, gold_ends, cluster_ids, ner_indices):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        subword_index = subword_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        ner_indices = ner_indices[sentence_offset:sentence_offset + max_training_sentences, :]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        if not len(gold_starts):
            return None

        return (tokens, context_word_emb, head_word_emb, char_index, subword_index, text_len, speaker_ids, genre, 
                is_training, gold_starts, gold_ends, cluster_ids, ner_indices)

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))
        same_span = tf.cast(tf.logical_and(same_start, same_end), dtype=tf.int32)
        is_gold_span = tf.reduce_sum(input_tensor=same_span, axis=0)
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), same_span)
        candidate_labels = tf.squeeze(candidate_labels, 0)
        return candidate_labels, is_gold_span

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.cast(is_training, dtype=tf.float32) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)
        antecedents_mask = antecedent_offsets >= 1
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0)
        fast_antecedent_scores += tf.math.log(tf.cast(antecedents_mask, dtype=tf.float32))
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets
        top_antecedents_mask = raw_top_antecedents >= 0
        top_antecedents = tf.maximum(raw_top_antecedents, 0)

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents)
        top_fast_antecedent_scores += tf.math.log(tf.cast(top_antecedents_mask, dtype=tf.float32))
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def compute_b3_loss(self, p_m_entity, gold_m_entity, beta=2.0):
        gold_entities = tf.reduce_sum(input_tensor=gold_m_entity, axis=0) > 1.2
        gold_entities = tf.cond(pred=tf.reduce_any(input_tensor=gold_entities),
                                true_fn=lambda: gold_entities,
                                false_fn=lambda: tf.reduce_sum(input_tensor=gold_m_entity, axis=0) > 0)
        k = tf.shape(input=p_m_entity)[0]
        sys_m_e = tf.one_hot(tf.argmax(input=p_m_entity, axis=1), k)
        sys_entities = tf.reduce_sum(input_tensor=sys_m_e, axis=0) > 1.2
        sys_entities = tf.cond(pred=tf.reduce_any(input_tensor=sys_entities),
                               true_fn=lambda: sys_entities,
                               false_fn=lambda: tf.reduce_sum(input_tensor=sys_m_e, axis=0) > 0)

        gold_entity_filter = tf.reshape(tf.compat.v1.where(gold_entities), [-1])
        gold_cluster = tf.gather(tf.transpose(a=gold_m_entity), gold_entity_filter)

        sys_entity_filter, merge = tf.cond(pred=tf.reduce_any(input_tensor=sys_entities & gold_entities),
                                          true_fn=lambda: (tf.reshape(tf.compat.v1.where(sys_entities), [-1]), tf.constant(0)),
                                          false_fn=lambda: (
                                              tf.reshape(tf.compat.v1.where(sys_entities | gold_entities), [-1]), tf.constant(1)))
        system_cluster = tf.gather(tf.transpose(a=p_m_entity), sys_entity_filter)

        gold_sys_intersect = tf.pow(tf.matmul(gold_cluster, system_cluster, transpose_b=True), 2)
        r_num = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=gold_sys_intersect, axis=1) / tf.reduce_sum(input_tensor=gold_cluster, axis=1))
        r_den = tf.reduce_sum(input_tensor=gold_cluster)
        recall = tf.reshape(r_num / r_den, [])

        sys_gold_intersection = tf.transpose(a=gold_sys_intersect)
        p_num = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=sys_gold_intersection, axis=1) / tf.reduce_sum(input_tensor=system_cluster, axis=1))
        p_den = tf.reduce_sum(input_tensor=system_cluster)
        prec = tf.reshape(p_num / p_den, [])

        beta_2 = beta ** 2
        f_beta = (1 + beta_2) * prec * recall / (beta_2 * prec + recall)

        lost = 1.-f_beta
        return lost

    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, char_index, subword_index, text_len,
                                 speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, ner_indices):
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(input=context_word_emb)[0]
        max_sentence_length = tf.shape(input=context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]
        


        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.compat.v1.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                char_index)
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                            util.shape(char_emb, 3)])

            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["char_lstm_size"] // 2)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["char_lstm_size"] // 2)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, flattened_char_emb, dtype=tf.float32)
            flattened_aggregated_char_emb = tf.concat([output_fw[:, -1, :], output_bw[:, -1, :]], axis=-1)

            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                                  util.shape(flattened_aggregated_char_emb, 1)])
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        context_emb = tf.concat(context_emb_list, 2)
        head_emb = tf.concat(head_emb_list, 2)
        context_emb = tf.nn.dropout(context_emb, rate=1 - (self.lexical_dropout))
        head_emb = tf.nn.dropout(head_emb, rate=1 - (self.lexical_dropout))

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
        
        if self.config.get("use_pre_lstm_attention", True):
            num_heads = self.config.get("num_attention_heads", 4)
            attention_size = self.config.get("attention_size", 64)
            with tf.compat.v1.variable_scope("pre_lstm_attention"):
                context_emb = self.multi_head_attention(
                    context_emb, context_emb, context_emb,
                    num_heads=num_heads,
                    attention_size=attention_size,
                    dropout_rate=self.dropout
                )

        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)
        num_words = util.shape(context_outputs, 0)

        # === NER Prediction Module ===
        flattened_text_len_mask = self.flatten_emb_by_sentence(text_len_mask, text_len_mask)
        flattened_ner_indices = self.flatten_emb_by_sentence(ner_indices, text_len_mask)
        
        # Predict NER tags
        predicted_ner, self.ner_loss = self.build_ner_module(
            context_outputs,
            text_len_mask,
            flattened_ner_indices,
            is_training
        )
        
        # === Candidate Generation with Predicted NER ===
        ner_mask = predicted_ner > 0  # 0 = "no entity"
        ner_starts = tf.reshape(tf.where(ner_mask), [-1])
        ner_ends = ner_starts  # Single-word spans (can expand if needed)

        genre_emb = tf.gather(tf.compat.v1.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                                genre)

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                    [1, max_sentence_length])
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)

        # Generate regular candidate spans
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                    [1, self.max_span_width])
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0)
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                        candidate_starts)
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                            num_words - 1))
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                                candidate_end_sentence_indices))
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])
        regular_candidate_starts = tf.boolean_mask(tensor=tf.reshape(candidate_starts, [-1]),
                                            mask=flattened_candidate_mask)
        regular_candidate_ends = tf.boolean_mask(tensor=tf.reshape(candidate_ends, [-1]), mask=flattened_candidate_mask)
        candidate_sentence_indices = tf.boolean_mask(tensor=tf.reshape(candidate_start_sentence_indices, [-1]),
                                                        mask=flattened_candidate_mask)

        # Combine regular and NER candidates
        regular_candidate_starts = tf.cast(regular_candidate_starts, tf.int32)
        ner_starts = tf.cast(ner_starts, tf.int32)
        candidate_starts = tf.concat([regular_candidate_starts, ner_starts], axis=0)
        regular_candidate_ends = tf.cast(regular_candidate_ends, tf.int32)
        ner_ends = tf.cast(ner_ends, tf.int32)
        candidate_ends = tf.concat([regular_candidate_ends, ner_ends], axis=0)
        
        # Remove duplicates and sort
        max_val_for_encoding = tf.cast(num_words, tf.int32)
        candidate_starts_int32 = tf.cast(candidate_starts, tf.int32)
        candidate_ends_int32 = tf.cast(candidate_ends, tf.int32)
        unique_span_scalars = candidate_starts_int32 * (max_val_for_encoding + 1) + candidate_ends_int32
        unique_scalars, unique_idx = tf.unique(unique_span_scalars)
        candidate_starts = unique_scalars // (max_val_for_encoding + 1)
        candidate_ends = unique_scalars % (max_val_for_encoding + 1)

        candidate_cluster_ids, candidate_is_gold = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                                                cluster_ids)

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts,
                                                candidate_ends)
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb)
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)

        if self.config['mention_loss']:
            candidate_mention_scores_for_pruning = self.get_mention_scores(candidate_span_emb,
                                                                            "mention_scores_for_loss")
            candidate_mention_scores_for_pruning = tf.squeeze(candidate_mention_scores_for_pruning, 1)
        else:
            candidate_mention_scores_for_pruning = candidate_mention_scores

        if self.config['use_gold_mentions']:
            k = tf.cast(tf.shape(input=gold_starts)[0], dtype=tf.int32)
            top_span_indices = coref_ops.extract_spans(candidate_starts,
                                                        candidate_ends,
                                                        gold_starts,
                                                        gold_ends,
                                                        True)
        else:
            k = tf.cast(tf.floor(tf.cast(tf.shape(input=context_outputs)[0], dtype=tf.float32) * self.config["top_span_ratio"]), dtype=tf.int32)
            top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores_for_pruning, 0),
                                                        tf.expand_dims(candidate_starts, 0),
                                                        tf.expand_dims(candidate_ends, 0),
                                                        tf.expand_dims(k, 0),
                                                        util.shape(context_outputs, 0),
                                                        True)

        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)

        top_span_starts = tf.gather(candidate_starts, top_span_indices)
        top_span_ends = tf.gather(candidate_ends, top_span_indices)
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)
        self.top_span_mention_scores = top_span_mention_scores
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)
        top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)

        cluster_id_to_first_mention_id = tf.math.unsorted_segment_min(tf.range(k), top_span_cluster_ids, k)
        mention_id_to_first_mention_id = tf.gather(cluster_id_to_first_mention_id, top_span_cluster_ids)
        valid_cluster_ids = tf.cast(top_span_cluster_ids > 0, dtype=tf.int32)
        mention_id_to_first_mention_id = (mention_id_to_first_mention_id * valid_cluster_ids +
                                            tf.range(k) * (1 - valid_cluster_ids))
        gold_entity_matrix = tf.one_hot(mention_id_to_first_mention_id, k)

        c = tf.minimum(self.config["max_top_antecedents"], k)

        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
                top_span_emb, top_span_mention_scores, c)

        dummy_scores = tf.zeros([k, 1])
        top_refined_emb = None
        top_antecedent_scores = tf.concat([dummy_scores, top_fast_antecedent_scores], 1)
        
        for i in range(self.config["coref_depth"]):
            with tf.compat.v1.variable_scope("coref_layer_{}".format(1 if self.config["refinement_sharing"] else i),
                                                reuse=tf.compat.v1.AUTO_REUSE):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)
                
                if self.config.get("use_multihead_attention", True):
                    num_heads = self.config.get("num_attention_heads", 4)
                    attention_size = self.config.get("attention_size", 64)
                    
                    query = tf.expand_dims(top_span_emb, 1)
                    keys = top_antecedent_emb
                    values = top_antecedent_emb
                    
                    attention_output = self.multi_head_attention(
                        query, keys, values, 
                        num_heads=num_heads,
                        attention_size=attention_size,
                        dropout_rate=self.dropout,
                        scope="antecedent_attention"
                    )
                    attention_output = tf.squeeze(attention_output, 1)
                    slow_antecedent_scores = self.get_slow_antecedent_scores(
                        attention_output, top_antecedents, top_antecedent_emb, 
                        top_antecedent_offsets, top_span_speaker_ids, genre_emb
                    )
                else:
                    slow_antecedent_scores = self.get_slow_antecedent_scores(
                        top_span_emb, top_antecedents, top_antecedent_emb, 
                        top_antecedent_offsets, top_span_speaker_ids, genre_emb
                    )
                
                top_antecedent_scores = top_fast_antecedent_scores + slow_antecedent_scores
                top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)
                top_antecedent_weights = tf.nn.softmax(top_antecedent_scores)

                mention_indices = tf.tile(tf.expand_dims(tf.range(k, dtype=top_antecedents.dtype), 1), [1, c + 1])
                antecedent_indices = tf.concat([tf.expand_dims(tf.range(k, dtype=top_antecedents.dtype), 1), top_antecedents], axis=-1)
                antecedent_matrix_scatter_indices = tf.stack([mention_indices, antecedent_indices], axis=-1)
                antecedent_matrix = tf.scatter_nd(antecedent_matrix_scatter_indices, top_antecedent_weights, [k, k])
                entity_matrix = util.compute_p_m_entity(antecedent_matrix, k)

                # Create TensorBoard summaries
                with tf.name_scope('coref_visualization'):
                    entity_heatmap = tf.expand_dims(tf.expand_dims(entity_matrix, 0), 3)
                    self.entity_matrix_heatmap = tf.summary.image('entity_matrix_heatmap', entity_heatmap, max_outputs=1)
                    self.entity_prob_hist = tf.summary.histogram('entity_probabilities', entity_matrix)
                    self.avg_entity_prob = tf.summary.scalar('avg_entity_prob', tf.reduce_mean(entity_matrix))
                
                if self.config['entity_average']:
                    entity_matrix = entity_matrix / (
                        tf.reduce_sum(input_tensor=entity_matrix, axis=0, keepdims=True) + 1e-6)

                if self.config["antecedent_averaging"]:
                    top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1)
                    top_aa_emb = tf.reduce_sum(input_tensor=tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                axis=1)
                else:
                    top_aa_emb = top_span_emb

                if self.config["entity_equalization"]:
                    antecedent_mask = tf.cast(tf.sequence_mask(tf.range(k) + 1, k), dtype=tf.float32)
                    antecedent_mask = tf.expand_dims(antecedent_mask, 2)
                    entity_matrix_per_timestep = tf.expand_dims(entity_matrix, 0) * antecedent_mask
                    entity_emb_per_timestep = tf.tensordot(entity_matrix_per_timestep, top_aa_emb, [[1], [0]])
                    mention_entity_emb_per_timestep = tf.tensordot(entity_matrix, entity_emb_per_timestep, [[1], [1]])
                    indices = tf.tile(tf.expand_dims(tf.range(k), 1), [1, 2])
                    top_ee_emb = tf.gather_nd(mention_entity_emb_per_timestep, indices)
                else:
                    top_ee_emb = top_aa_emb

                top_refined_emb = top_ee_emb

            with tf.compat.v1.variable_scope("f", reuse=tf.compat.v1.AUTO_REUSE):
                f = tf.sigmoid(util.projection(tf.concat([top_span_emb, top_refined_emb], 1),
                                                util.shape(top_span_emb, -1)))
                top_span_emb = f * top_refined_emb + (1 - f) * top_span_emb
        
        self.b3_loss = self.compute_b3_loss(entity_matrix, gold_entity_matrix) * 10.
        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)
        top_antecedent_cluster_ids += tf.cast(tf.math.log(tf.cast(top_antecedents_mask, dtype=tf.float32)), dtype=tf.int32)
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)
        dummy_labels = tf.logical_not(tf.reduce_any(input_tensor=pairwise_labels, axis=1, keepdims=True))
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)
        self.antecedent_loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)
        self.antecedent_loss = tf.reduce_sum(input_tensor=self.antecedent_loss)

        gold_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.expand_dims(candidate_is_gold, 0),
                                                                tf.expand_dims(candidate_mention_scores_for_pruning, 0),
                                                                reduction='none')[0]
        positive_gold_losses = tf.cond(pred=tf.reduce_any(input_tensor=tf.equal(candidate_is_gold, 1)),
                                        true_fn=lambda: tf.boolean_mask(tensor=gold_loss, mask=tf.equal(candidate_is_gold, 1)),
                                        false_fn=lambda: gold_loss)
        negative_gold_losses = tf.cond(pred=tf.reduce_any(input_tensor=tf.equal(candidate_is_gold, 0)),
                                        true_fn=lambda: tf.boolean_mask(tensor=gold_loss, mask=tf.equal(candidate_is_gold, 0)),
                                        false_fn=lambda: gold_loss)
        n_pos = tf.shape(input=positive_gold_losses)[0]
        n_neg = tf.minimum(n_pos * 10, tf.shape(input=negative_gold_losses)[0])
        negative_gold_losses, _ = tf.nn.top_k(negative_gold_losses, n_neg, sorted=False)
        ohem_gold_loss = tf.reduce_mean(input_tensor=tf.concat([positive_gold_losses, negative_gold_losses], axis=0))
        ohem_gold_loss = ohem_gold_loss * 100.
        self.mention_loss = ohem_gold_loss

        losses = []
        if self.config['b3_loss']:
            losses.append(self.b3_loss)
        if self.config['antecedent_loss']:
            losses.append(self.antecedent_loss)
        if self.config['mention_loss']:
            losses.append(self.mention_loss)
        
        # Add NER loss with configurable weight
        ner_loss_weight = self.config.get("ner_loss_weight", 0.1)
        losses.append(ner_loss_weight * self.ner_loss)
        
        # Add L2 regularization
        l2_lambda = self.config.get("l2_lambda", 0.01)  # Default regularization strength
        l2_regularization = l2_lambda * tf.add_n([
            tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'bias' not in v.name
        ])
        losses.append(l2_regularization)
        
        loss = tf.add_n(losses)

        # Merge all summaries
        self.entity_matrix_summaries = tf.summary.merge([
            self.entity_matrix_heatmap,
            self.entity_prob_hist,
            self.avg_entity_prob
        ])
        return [candidate_starts, candidate_ends, candidate_mention_scores, 
                top_span_starts, top_span_ends, top_antecedents, 
                top_antecedent_scores, predicted_ner], loss

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts

        if self.config["use_features"]:
            span_width_index = span_width - 1
            span_width_emb = tf.gather(
                tf.compat.v1.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)
            span_width_emb = tf.nn.dropout(span_width_emb, rate=1 - (self.dropout))
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1)
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)
            span_text_emb = tf.gather(head_emb, span_indices)
            with tf.compat.v1.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)
                span_head_scores = tf.gather(self.head_scores, span_indices)
                span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2)
                span_head_scores += tf.math.log(span_mask)
                self.span_attention = tf.nn.softmax(span_head_scores, 1)
                span_head_emb = tf.reduce_sum(input_tensor=self.span_attention * span_text_emb, axis=1)
                span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)
        return span_emb

    def get_mention_scores(self, span_emb, name=None):
        with tf.compat.v1.variable_scope(name, "mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.math.log(tf.cast(antecedent_labels, dtype=tf.float32))
        marginalized_gold_scores = tf.reduce_logsumexp(input_tensor=gold_scores, axis=[1])
        log_norm = tf.reduce_logsumexp(input_tensor=antecedent_scores, axis=[1])
        return log_norm - marginalized_gold_scores

    def bucket_distance(self, distances):
        logspace_idx = tf.cast(tf.floor(tf.math.log(tf.cast(distances, dtype=tf.float32)) / math.log(2)), dtype=tf.int32) + 3
        use_identity = tf.cast(distances <= 4, dtype=tf.int32)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb_original, top_antecedent_offsets,
                                    top_span_speaker_ids, genre_emb):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)
            speaker_pair_emb = tf.gather(tf.compat.v1.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                            tf.cast(same_speaker, dtype=tf.int32))
            feature_emb_list.append(speaker_pair_emb)

        tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])
        feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)
            antecedent_distance_emb = tf.gather(
                tf.compat.v1.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)
        feature_emb = tf.nn.dropout(feature_emb, rate=1 - (self.dropout))

        from_tensor = tf.expand_dims(top_span_emb, 1)
        to_tensor = top_antecedent_emb_original

        # Project to_tensor to match the last dimension of from_tensor
        target_dimension = util.shape(from_tensor, -1)
        if util.shape(to_tensor, -1) != target_dimension:
            with tf.compat.v1.variable_scope("project_to_tensor_for_similarity"):
                to_tensor = util.projection(to_tensor, target_dimension)

        similarity_emb = from_tensor * to_tensor
        from_tensor = tf.tile(from_tensor, [1, c, 1])

        pair_emb = tf.concat([from_tensor, to_tensor, similarity_emb, feature_emb], 2)

        with tf.compat.v1.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                                self.dropout)
            slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)

        return slow_antecedent_scores

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(input=text_emb)[0]

        current_inputs = text_emb

        for layer in range(self.config["contextualization_layers"]):
            with tf.compat.v1.variable_scope("contextualization_layer_{}".format(layer)):
                 input_list = [current_inputs]

                 if self.config["use_features"]:
                    width = tf.shape(input=current_inputs)[1]
                    feature_indices = tf.tile(tf.expand_dims(tf.range(width), 0), [num_sentences, 1])
                    feature_indices = tf.minimum(feature_indices, self.config["feature_size"] - 1)
                    feature_indices = tf.expand_dims(feature_indices, 2)
                    feature_emb = tf.gather(tf.compat.v1.get_variable("feature_embeddings", [self.config["feature_size"], self.config["feature_size"]]), feature_indices)
                    feature_emb = tf.squeeze(feature_emb, 2)
                    input_list.append(feature_emb)

                 inputs = tf.concat(input_list, 2)
                 num_sentences = tf.shape(input=text_emb)[0]
                 with tf.compat.v1.variable_scope("fw_cell"):
                      lstm_cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
                 with tf.compat.v1.variable_scope("bw_cell"):
                      lstm_cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)

                 # Fix: Compute sequence_length from the mask to get the correct shape.
                 # This is the crucial part that fixes the assertion error.
                 sequence_lengths = tf.reduce_sum(tf.cast(text_len_mask, tf.int32), axis=1)

                 (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs,              sequence_length=sequence_lengths,  # Use the correctly shaped tensor
                     dtype=tf.float32)
                 current_inputs = tf.concat([output_fw, output_bw], 2)

        return self.flatten_emb_by_sentence(current_inputs, text_len_mask)    
    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(input=emb)[0]
        max_sentence_length = tf.shape(input=emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank  == 3:
            flattened_emb = tf.boolean_mask(tensor=tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)]),
                                                mask=tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))
        elif emb_rank == 2:
            flattened_emb = tf.boolean_mask(tensor=tf.reshape(emb, [num_sentences * max_sentence_length]),
                                                mask=tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return flattened_emb

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.compat.v1.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                rate=1 - (self.dropout))
        target_top_span_emb = tf.nn.dropout(top_span_emb, rate=1 - (self.dropout))
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents, include_singletons=True):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        if include_singletons or self.config['include_singletons']:
            for i, predicted_index in enumerate(predicted_antecedents):
              mention = (int(top_span_starts[i]), int(top_span_ends[i]))
              if mention not in mention_to_predicted:
                 predicted_cluster = len(predicted_clusters)
                 mention_to_predicted[mention] = predicted_cluster
                 predicted_clusters.append([mention])

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [x for x in (load_line(l) for l in f.readlines()) if x[0] is not None]
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False, pprint=False, test=False, outfpath=None):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()

        if outfpath:
          fh = open(outfpath, 'w')

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}

            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, predicted_ner = session.run(
                self.predictions, feed_dict=feed_dict)

            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)

            if pprint:
                tokens = util.flatten(example["sentences"])
                print("=== Predicted NER ===")
                for i, (token, tag_idx) in enumerate(zip(tokens, predicted_ner)):
                    tag = self.ner_types[tag_idx-1] if tag_idx > 0 else "O"
                    print(f"{token} ({tag})")
                
                print("\nGOLD CLUSTERS:")
                util.coref_pprint(tokens, example["clusters"])
                print("PREDICTED CLUSTERS:")
                util.coref_pprint(tokens, coref_predictions[example["doc_key"]])
                print('==================================================================')

            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

            if outfpath:
               predicted_clusters_sg, _ = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents, include_singletons=True)
               
               copy = { k: v for k, v in example.items() }
               copy['clusters'] = predicted_clusters_sg
               copy['predicted_ner'] = [self.ner_types[idx - 1] if 0 < idx <= len(self.ner_types) else "O" for idx in predicted_ner]
               #copy['predicted_ner'] = [self.ner_types[idx-1] if idx > 0 else "O" for idx in predicted_ner]
               fh.write(json.dumps(copy)+"\n")

        if outfpath:
          fh.close()

        mention_p, mention_r, mention_f = metrics.get_prf_mentions_for_all_documents([e[1] for e in self.eval_data], coref_predictions)

        summary_dict = {}

        p, r, f = coref_evaluator.get_prf()
        average_f1 = f * 100
        summary_dict["Average F1 (py)"] = average_f1
        print("Average F1 (py): {:.2f}%".format(average_f1))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        average_mention_f1 = mention_f * 100
        summary_dict["Average mention F1 (py)"] = average_mention_f1
        print("Average mention F1 (py): {:.2f}%".format(average_mention_f1))
        summary_dict["Average mention precision (py)"] = mention_p
        print("Average mention precision (py): {:.2f}%".format(mention_p * 100))
        summary_dict["Average mention recall (py)"] = mention_r
        print("Average mention recall (py): {:.2f}%".format(mention_r * 100))

        return util.make_summary(summary_dict), average_f1, average_mention_f1
