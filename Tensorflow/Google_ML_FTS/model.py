import tensorflow as tf
class featuresAndAnswers:
    def __init__(self, features, answers):
        self.features = features
        self.answers = answers
class trainingAndTest():
    def __init__(self, features, answers ,percentage):
        training_size = int(len(features) * percentage)
        self.traning = featuresAndAnswers(features[:training_size],answers[:training_size])
        self.test = featuresAndAnswers(features[training_size:],answers[training_size:])
        # N225_1 とかいっぱい
        self.feature_type_count = len(features.columns)
        # positive nagativeの２つ
        self.answer_type_count = len(answers.columns)
class Model:
    def __init__(self, features, answers, layers=[]):
        # 80%をトレーニングに使う
        self.data = trainingAndTest(features, answers, 0.8)
        # placeholderは変数みたいなもん
        self.real_answer = tf.placeholder(tf.float32,  shape=(None, self.data.answer_type_count))
        self.feature = tf.placeholder(tf.float32, shape=(None, self.data.feature_type_count))
        self.model = self.createTfModel(layers)
        # 目標値との誤差 reduce_sumで全部足しちゃう
        cost = -tf.reduce_sum(self.real_answer*tf.log(self.model))
        # 最適化のアルゴリズム。アダムは評価が高いらしいほかにも10個位tensorflow api にある
        self.step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # 正答率の算出 いつもおんなじ？
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.real_answer, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def createTfModel(self, layers):
        hidden_layer = None
        loop_count_max = len(layers)+1
        for loop_count in range(0,loop_count_max):
            if loop_count is len(layers) :
                #最後は与えられた答えの数 positive negativeの2になる
                need_answer_count = self.data.answer_type_count
            else :
                #隠れ層がある場合はその層のニューロンの数？の答えをだすのね
                need_answer_count = layers[loop_count]
            if loop_count is 0 :
                # 最初は与えられた特徴なのね
                feature = self.feature
                feature_type_count = self.data.feature_type_count
            else :
                # 隠れ層がある場合は隠れ層が学習する特徴のデータになるんだ！
                feature = hidden_layer
                feature_type_count = layers[loop_count-1]
            # truncated_normal Tensorを正規分布かつ標準偏差0.0001の２倍までのランダムな値で初期化する
            weights = tf.Variable(tf.truncated_normal([feature_type_count, need_answer_count], stddev=0.0001))
            # バイアス
            biases = tf.Variable(tf.ones([need_answer_count]))
            # matmulは掛け算feature * weights
            logits = tf.matmul(feature, weights) + biases
            if loop_count is loop_count_max - 1  :
                # 最後はsoftmax
                return tf.nn.softmax(logits)
            else :
                # reluはRectified Linear Unit, Rectifier, 正規化線形関数だそうです。
                hidden_layer = tf.nn.relu(logits)
    def train(self,count=30000,print_count=5):
        feed_dict = {
               self.feature: self.data.traning.features,
               self.real_answer: self.data.traning.answers
            }
        for i in range(1, count+1):
            # feed_dictからself.stepを評価
            self.session.run(self.step,feed_dict)
            if i % (count/print_count) == 0:
                # feed_dictからself.accuracyを評価してるのでaccuracyが返ってくる
                print( i, self.session.run(self.accuracy,feed_dict))
    def test(self):
        predictions = tf.argmax(self.model, 1)
        real_answers = tf.argmax(self.real_answer, 1)
        count_correct_answer = tf.reduce_sum(
          # booleanをfloatに。trueが１になるのでしょう。
          tf.cast(
            # booleanがもどる
            tf.equal(real_answers, predictions),
            tf.float32
          )
        )
        correct_answer_count = self.session.run(
            count_correct_answer,
            {self.feature: self.data.test.features,
             self.real_answer: self.data.test.answers}
          )
        return  correct_answer_count/len(self.data.test.answers)
