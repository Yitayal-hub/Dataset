#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import util
import anapho_model as cm

if __name__ == "__main__":
    args = util.get_args()
    config = util.initialize_from_env(args.experiment, args.logdir)

    # Ensure character-level BiLSTM parameters are in config
    if "char_embedding_size" not in config:
        raise ValueError("char_embedding_size must be specified in config.")
    if "char_lstm_size" not in config:
        raise ValueError("char_lstm_size must be specified in config.")
    if "char_vocab_path" not in config:
        raise ValueError("char_vocab_path must be specified in config.")

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]
    max_step = config["max_step"]
    model = cm.AnaphoModel(config)

    print('# parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))
    saver = tf.compat.v1.train.Saver()
    initial_step = 0

    log_dir = config["log_dir"]
    writer = tf.compat.v1.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            initial_step = session.run(model.global_step)

        initial_time = time.time()
        while True:
            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
            accumulated_loss += tf_loss

            if tf_global_step % report_frequency == 0:
                steps_per_second = (tf_global_step - initial_step) / (time.time() - initial_time)

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step,
                                                                    average_loss,
                                                                    steps_per_second))
                writer.add_summary(util.make_summary({"loss": average_loss,
                                                        "learning_rate": session.run(model.learning_rate)}), tf_global_step)
                accumulated_loss = 0.0
                initial_time = time.time()
                initial_step = tf_global_step

            if tf_global_step % config['save_frequency'] == 0:
                saver.save(session, os.path.join(log_dir, "saved_model"), global_step=tf_global_step)

            if tf_global_step % eval_frequency == 0:
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                eval_summary, eval_f1, mention_f1 = model.evaluate(session)

                if config["eval_for_mentions"]:
                    if mention_f1 > max_f1:
                        max_f1 = mention_f1
                        util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
                else:
                    if eval_f1 > max_f1:
                        max_f1 = eval_f1
                        util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

                writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

                msg = "[{}] eval_f1 (anapho)={:.4f}, eval_f1 (mention)={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, mention_f1, max_f1)
                print(msg)

            if max_step > 0 and tf_global_step >= max_step:
                break
