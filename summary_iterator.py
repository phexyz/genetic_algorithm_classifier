import tensorflow as tf


for e in tf.train.summary_iterator("./events.out.tfevents.1532024689.21c7ad2c00de"):
    for v in e.summary.value:
        if v.tag == 'loss' :
            print(v.simple_value)