FROM docker pull pytorch/torchserve:0.3.0-cpu

ADD mnist_cnn.pt /tmp/
ADD mnist_handler.py /tmp/
ADD mnist.py /tmp/

RUN torch-model-archiver --model-name mnist --version 1.0 --model-file /tmp/mnist.py --serialized-file /tmp/mnist_cnn.pt --handler /tmp/mnist_handler.py && \
     mkdir -p /tmp/models && \
     mv mnist.mar /tmp/models

CMD ["torchserve", "--start", "--model-store", "/tmp/models", "--models", "mnist=mnist.mar"]