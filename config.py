class Config(object):
    training_config = {
        "tat_path": "./model/tat/",
        "baseline_epoch": 30,
        "tav_path": "./model/tav/",
        "mta_path": "./model/mta/"
    }

    train_data_path = [
        # please use your own preprocessed data. 
    ]
    test_data_path = [
        # please use your own preprocessed data. 
    ]

    generator_config = {
        "embedding_size": 200,  #
        "hidden_size": 512,  #
        "max_len": 100,
        "start_token": 0,
        "eos_token": 1,
        "batch_size": 128,
        "vocab_size": 50004,
        "grad_norm": 10,
        "topic_num": 5,
        "is_training": True,
        "keep_prob": .5,
        "norm_init": 0.05,
        "normal_std": 1,
        "learning_rate": 1e-3,
        "beam_width": 5,
        "mem_num": 60
    }
    data_dir = 'Data/'
    vec_file = './vec.txt'
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10 #gradient clipping
    num_layers = 2
    num_steps = 1000 #this value is one more than max number of words in sentence
    hidden_size = 20
    word_embedding_size = 100
    max_epoch = 20
    max_max_epoch = 22
    keep_prob = 0.5 #The probability that each element is kept through dropout layer
    lr_decay = 1.0
    batch_size = 16
    vocab_size = 7187
    num_keywords = 5
    save_freq = 10 #The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_path = './Model_News' #the path of model that need to save or load
    
    # parameter for generation
    len_of_generation = 16 #The number of characters by generated
    save_time = 20 #load save_time saved models
    is_sample = True #true means using sample, if not using argmax
    BeamSize = 2


