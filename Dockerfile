# docker-keras - Keras in Docker with Python 3 and TensorFlow on GPU

FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN pip3 install keras==2.1.6 pandas


# quick test and dump package lists
RUN python3 -c "import tensorflow; print(tensorflow.__version__)" \
 && dpkg-query -l > /dpkg-query-l.txt \
 && pip3 freeze > /pip3-freeze.txt

WORKDIR /srv/