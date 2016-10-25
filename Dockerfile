FROM ubuntu:14.04
MAINTAINER Melchior.fracas@wanadoo.fr

RUN apt-get install -y libfreetype6-dev && \
    apt-get install -y libglib2.0-0 \
                       libxext6 \
                       libsm6 \
                       libxrender1 \
                       libblas-dev \
                       liblapack-dev \
                       gfortran \
                       libfontconfig1 --fix-missing

RUN apt-get install tar \
                    git \
                    curl \
                    nano \
                    wget \
                    dialog \
                    net-tools \
                    build-essential

RUN apt-get install -y python \
                       python-dev \
                       python-distribute \
                       python-pip

RUN pip install matplotlib \
                seaborn \
                pandas \
                numpy \
                scipy \
                sklearn \
                python-dateutil \
                gensim

ENTRYPOINT ["python"]
CMD ["my_script.py"]
