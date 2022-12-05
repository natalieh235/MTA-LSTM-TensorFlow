import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from Config import Config
from util import *
from collections import defaultdict
from sc_lstm_cell import SCLSTM, ActionWrapper, SC_DropoutWrapper


class SCLSTM_Model:
    def __init__(self, config):
        # configuration
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.max_len = config["max_len"]

        self.learning_rate = config["learning_rate"]
        self.sos_token = config["start_token"]
        self.eos_token = config["eos_token"]
        self.batch_size = config["batch_size"]
        self.sequence_lengths = [self.max_len] * self.batch_size

        self.vocab_size = config["vocab_size"]
        self.vocab_dict = config["vocab_dict"]
        self.vocab_size = 50000 + 4  # GO EOS UNK PAD
        # len(self.vocab_dict)

        self.grad_norm = config["grad_norm"]
        self.topic_num = config["topic_num"]
        self.training_flag = config["is_training"]
        self.keep_prob = config["keep_prob"]
        self.norm_init = config["norm_init"]
        self.normal_std = config["normal_std"]
        self.pretrain_wv = config["pretrain_wv"]
        self.attention_size = config["attention_size"]
        self.topic_size = config["topic_size"]
        self.refers = None
        self.rand_uni_init = tf.random_uniform_initializer(-self.norm_init, self.norm_init,
                                                           seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.normal_std)
        self.sm = SmoothingFunction()

    def build_placeholder(self):
        # placeholder
        self.d_act = tf.placeholder(tf.float32, [self.batch_size, self.topic_size], name="action_vector")

        self.target_input = tf.placeholder(tf.int32, [self.batch_size, None], name="target_index")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        # self.target_mask = tf.placeholder(tf.float32, [self.batch_size, self.max_len], name="target_mask")

    def build_graph(self):
        print("building generator graph...")
        with tf.variable_scope("seq2seq"):
            with tf.variable_scope("embedding"):
                self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32,
                                                 trainable=True, initializer=tf.constant_initializer(self.pretrain_wv))

            # we need one hot topic vector -> src_lbl_oh
            dtype = tf.float32
            with tf.variable_scope("decoder"):
                def _get_cell(_num_units):
                    cell = SCLSTM(self.topic_size, self.hidden_size, dtype=dtype)
                    # current the droppout is not supported
                    if self.training_flag:
                        cell = SC_DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                    return cell

                # single layer
                self.wr = tf.get_variable('sc_wr', [self.embedding_size, self.topic_size], dtype=tf.float32,
                                          trainable=True,
                                          initializer=self.rand_uni_init)  # [word_embedding_size, topic_size ]
                self.hr = tf.get_variable('sc_hr', [self.hidden_size, self.topic_size], dtype=tf.float32,
                                          trainable=True, initializer=self.rand_uni_init)

                self.decoder_cell = _get_cell(self.hidden_size)
                self.initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.decoder_input_embedded = tf.nn.embedding_lookup(self.embedding, self.target_input)
                self.output_layer = layers_core.Dense(self.vocab_size, use_bias=False)

                # pre-train with targets #
                helper_pt = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_input_embedded,
                    sequence_length=self.sequence_lengths,
                    time_major=False,
                )
                masks = tf.sequence_mask(lengths=self.target_len,
                                         maxlen=self.max_len, dtype=tf.float32, name='masks')

                # NOTE: the cell must wrapped with ActionWrapper to adapt the api of seq2seq
                training_cell = ActionWrapper(self.decoder_cell, self.d_act, self.wr, self.hr)
                decoder_pt = tf.contrib.seq2seq.BasicDecoder(
                    cell=training_cell,
                    helper=helper_pt,
                    initial_state=self.initial_state,
                    output_layer=self.output_layer
                )

                outputs_pt, _final_state, sequence_lengths_pt = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder_pt,
                    output_time_major=False,
                    maximum_iterations=self.max_len,
                    swap_memory=True,
                    impute_finished=True
                )

                self.logits_pt = outputs_pt.rnn_output
                self.g_predictions = tf.nn.softmax(self.logits_pt)

                self.target_output = tf.placeholder(tf.int32, [None, None])

                self.pretrain_loss = tf.contrib.seq2seq.sequence_loss(
                    self.logits_pt,
                    self.target_output,
                    masks,
                    average_across_timesteps=True,
                    average_across_batch=True)

                self.global_step = tf.Variable(0, trainable=False)

                # gradient clipping
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(self.pretrain_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.grad_norm)
                self.pretrain_updates = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

            # infer
            helper_i = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding,
                tf.fill([self.batch_size], self.vocab_dict['<GO>']),
                end_token=self.vocab_dict['<EOS>']
            )
            # to avoid tensor in different loop, we create a same
            infer_cell = ActionWrapper(self.decoder_cell, self.d_act, self.wr, self.hr)
            decoder_i = tf.contrib.seq2seq.BasicDecoder(
                cell=infer_cell,
                helper=helper_i,
                initial_state=self.initial_state,
                output_layer=self.output_layer
            )

            outputs_i, _final_state_i, sequence_lengths_i = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder_i,
                output_time_major=False,
                maximum_iterations=self.max_len,
                swap_memory=True,
                impute_finished=True
            )

            sample_id = outputs_i.sample_id
            self.infer_tokens = tf.unstack(sample_id, axis=0)

        print("generator graph built successfully")

    def evaluate_bleu(self, sess, da):
        return sess.run(self.infer_tokens, feed_dict={self.d_act: da})

    def _padding(self, samples, max_len):
        batch_size = len(samples)
        samples_padded = np.zeros(shape=[batch_size, max_len], dtype=np.int32)  # == PAD
        for i, seq in enumerate(samples):
            for j, element in enumerate(seq):
                samples_padded[i, j] = element
        return samples_padded

    def run_pretrain_step(self, sess, batch_data):
        pre_train_fd = self._make_pretrain_feed_dict(batch_data)
        _, pretrain_loss = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict=pre_train_fd)
        return pretrain_loss

    def _make_pretrain_feed_dict(self, batch_data):
        da, target_input, target_len = batch_data
        # , target_mask
        target_padded, _ = self._pad_input_data(target_input)
        target_output = self._pad_target_data(target_input)
        return {self.d_act: da,
                self.target_input: target_padded,
                self.target_output: target_output,
                self.target_len: target_len
                }
        # self.target_mask: target_mask}

    def _get_weights(self, lengths):
        x_len = len(lengths)
        max_l = self.max_len
        ans = np.zeros((x_len, max_l))
        for ll in range(x_len):
            kk = lengths[ll] - 1
            for jj in range(kk):
                ans[ll][jj] = 1 / float(kk)
        return ans

    def _pad_input_data(self, x):
        max_l = self.max_len
        go_id = self.vocab_dict['<GO>']
        end_id = self.vocab_dict['<EOS>']
        x_len = len(x)
        ans = np.zeros((x_len, max_l), dtype=int)
        ans_lengths = []
        for i in range(x_len):
            ans[i][0] = go_id
            jj = min(len(x[i]), self.max_len - 2)
            for j in range(jj):
                ans[i][j + 1] = x[i][j]
            ans[i][jj + 1] = end_id
            ans_lengths.append(jj + 2)
        return ans, ans_lengths

    def _pad_target_data(self, x):
        max_l = self.max_len
        end_id = self.vocab_dict['<EOS>']
        x_len = len(x)
        ans = np.zeros((x_len, max_l), dtype=int)
        for i in range(x_len):
            jj = min(len(x[i]), max_l - 1)
            for j in range(jj):
                ans[i][j] = x[i][j]
            ans[i][jj] = end_id
        return ans

    def _pad_topic(self, x):
        max_num = self.topic_num  # 5
        size = len(x)
        ans = np.zeros((size, max_num), dtype=int)
        for i in range(size):
            true_len = min(len(x[i]), max_num)
            for j in range(true_len):
                ans[i][j] = x[i][j]
        return ans

    @staticmethod
    def restore(sess, saver, path):
        saver.restore(sess, save_path=path)
        print("load model successfully")

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print("model saved at %s" % save_path)




if __name__ == '__main__':
    config = Config().generator_config
    config["vocab_dict"] = np.load("path_to_vocab_dict").item()
    config["pretrain_wv"] = np.load("path_to_pretrain_wv")
    G = SCLSTM_Model(config)
    G.build_placeholder()
    G.build_graph()
