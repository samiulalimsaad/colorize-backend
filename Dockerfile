
FROM openfabric/openfabric-pyenv:0.1.6-3.8


RUN mkdir image-coloring

WORKDIR /image-coloring

COPY . .

RUN poetry install

EXPOSE 5000

CMD ["sh","start.sh"]