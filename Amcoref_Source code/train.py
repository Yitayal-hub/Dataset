#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import threading
from datetime import datetime

import numpy as np
import tensorflow as tf
import util
import coref_model as cm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # Log to file
        logging.StreamHandler()              # Log to console
    ]
)
logger = logging.getLogger(__name__)

def validate_config(config):
    """Validate the configuration dictionary for required parameters."""
    required_configs = [
        "char_embedding_size", 
        "char_lstm_size", 
        "char_vocab_path",
        "report_frequency",
        "eval_frequency",
        "max_step",
        "log_dir",
        "save_frequency"
    ]
    for param in required_configs:
        if param not in config:
            raise ValueError(f"Required config parameter '{param}' is missing.")

def setup_tensorboard(log_dir):
    """Set up TensorBoard summary writer."""
    os.makedirs(log_dir, exist_ok=True)
    return tf.compat.v1.summary.FileWriter(log_dir, flush_secs=20)

def initialize_training(session, model, saver, log_dir):
    """Initialize training session and restore from checkpoint if available."""
    session.run(tf.compat.v1.global_variables_initializer())
    model.start_enqueue_thread(session)
    
    initial_step = 0
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logger.info(f"Restoring from checkpoint: {ckpt.model_checkpoint_path}")
        saver.restore(session, ckpt.model_checkpoint_path)
        initial_step = session.run(model.global_step)
        logger.info(f"Resuming training from step {initial_step}")
    
    return initial_step

def training_loop(session, model, saver, writer, initial_step, config):
    """Main training loop with periodic evaluation and checkpointing."""
    accumulated_loss = 0.0
    max_f1 = 0
    best_step = 0
    start_time = initial_time = time.time()
    tf_global_step = initial_step  # Initialize tf_global_step
    
    try:
        while True:
            # Run training step - only fetch summaries periodically to reduce overhead
            if tf_global_step % config['report_frequency'] == 0:
                tf_loss, tf_global_step, _, summaries = session.run(
                    [model.loss, model.global_step, model.train_op, model.entity_matrix_summaries]
                )
                if summaries is not None:
                    writer.add_summary(summaries, tf_global_step)
            else:
                tf_loss, tf_global_step, _ = session.run(
                    [model.loss, model.global_step, model.train_op]
                )
            accumulated_loss += tf_loss

            # Periodic reporting
            if tf_global_step % config['report_frequency'] == 0:
                average_loss = accumulated_loss / config['report_frequency']
                steps_per_second = (tf_global_step - initial_step) / (time.time() - initial_time)
                
                logger.info(
                    f"[Step {tf_global_step}] loss={average_loss:.2f}, "
                    f"steps/s={steps_per_second:.2f}, "
                    f"elapsed={time.time()-start_time:.2f}s"
                )
                
                # Add standard training metrics to TensorBoard
                writer.add_summary(
                    util.make_summary({
                        "loss": average_loss,
                        "learning_rate": session.run(model.learning_rate)
                    }), 
                    tf_global_step
                )
                accumulated_loss = 0.0
                initial_time = time.time()
                initial_step = tf_global_step

            # Periodic checkpointing
            if tf_global_step % config['save_frequency'] == 0:
                save_path = saver.save(session, os.path.join(config['log_dir'], "saved_model"), global_step=tf_global_step)
                logger.info(f"Model checkpoint saved to {save_path}")

            # Periodic evaluation
            if tf_global_step % config['eval_frequency'] == 0:
                saver.save(session, os.path.join(config['log_dir'], "model"), global_step=tf_global_step)
                try:
                    eval_summary, eval_f1, mention_f1 = model.evaluate(session)

                    # Update best model
                    current_f1 = mention_f1 if config.get("eval_for_mentions", False) else eval_f1
                    if current_f1 > max_f1:
                        max_f1 = current_f1
                        best_step = tf_global_step
                        util.copy_checkpoint(
                            os.path.join(config['log_dir'], f"model-{tf_global_step}"), 
                            os.path.join(config['log_dir'], "model.max.ckpt")
                        )
                        logger.info(f"New best model at step {tf_global_step} with F1={max_f1:.4f}")

                    # Add evaluation metrics to TensorBoard
                    writer.add_summary(eval_summary, tf_global_step)
                    writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

                    logger.info(
                        f"[Evaluation at step {tf_global_step}] "
                        f"coref_f1={eval_f1:.4f}, "
                        f"mention_f1={mention_f1:.4f}, "
                        f"max_f1={max_f1:.4f} (from step {best_step})"
                    )
                except Exception as e:
                    logger.error(f"Evaluation failed at step {tf_global_step}: {str(e)}")
                    # Continue training even if evaluation fails
                    continue

            # Termination condition
            if config['max_step'] > 0 and tf_global_step >= config['max_step']:
                logger.info(f"Training completed after reaching max_step {config['max_step']}")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
    finally:
        # Proper cleanup
        model.close()
        return max_f1, best_step
if __name__ == "__main__":
    args = util.get_args()
    config = util.initialize_from_env(args.experiment, args.logdir)
    
    # Validate configuration
    validate_config(config)
    
    logger.info(f"Starting training with config: {config}")
    if 'subword_vocab_path' in config:
        logger.info(f"Subword vocab path: {config['subword_vocab_path']}")

    # Initialize model and log parameters
    model = cm.CorefModel(config)
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    logger.info(f'Total trainable parameters: {total_params:,}')

    # Set up training components
    saver = tf.compat.v1.train.Saver(
        max_to_keep=5,  # Keep only 5 most recent checkpoints
        save_relative_paths=True
    )
    writer = setup_tensorboard(config["log_dir"])

    # Configure session options for better performance
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True

    # Run training session with proper cleanup
    try:
        with tf.compat.v1.Session(config=session_config) as session:
            initial_step = initialize_training(session, model, saver, config["log_dir"])
            
            # Write the graph to TensorBoard
            writer.add_graph(session.graph)
            
            max_f1, best_step = training_loop(session, model, saver, writer, initial_step, config)
            
            writer.close()
            logger.info(f"Training session ended. Best F1 score: {max_f1:.4f} at step {best_step}")
    except Exception as e:
        logger.error(f"Fatal error during training: {str(e)}", exc_info=True)
        raise
