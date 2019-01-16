import tensorflow as tf

class BaseNet(object):
    """Base class for Net. It uses template to manage variables"""
    def __init__(self, name="BaseNet"):
        self.name = name

        # wrapping network into a template. It is useful for parameters sharing
        #self.template = tf.make_template(name, self.call, False)

    # def __getattr__(self, name):
    #     # make this class able to access template's method
    #     return self.template.__getattribute__(name)

    # def __call__(self, *args, **kwargs):
    #     call template
    #     return self.template(*args, **kwargs)

    def __call__(self, images, is_training, **kwargs):
        return self.call(images, is_training, **kwargs)


    def call(self, images, is_training, **kwargs):
        raise NotImplementedError("This method is not implemented.")