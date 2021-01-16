FROM python:3.7
RUN pip install uwsgi supervisor

ENV HOME=/app
COPY requirements.txt ${HOME}/
RUN pip install --upgrade pip
RUN pip install -r ${HOME}/requirements.txt --no-cache-dir --default-timeout=100

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

# mount the checkpoints dir

WORKDIR ${HOME}

# RUN pip install git+https://github.com/heartexlabs/label-studio.git@master
RUN pip install label-studio

COPY ./ ${HOME}/img_sgm_ml
COPY ./_wsgi.py ${HOME}/_wsgi.py
COPY ./supervisord.conf ${HOME}/supervisord.conf
COPY ./uwsgi.ini ${HOME}/uwsgi.ini

EXPOSE 9090
CMD ["supervisord"]
