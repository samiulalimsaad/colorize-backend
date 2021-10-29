
FROM test:sdk


RUN mkdir image-coloring

WORKDIR /image-coloring

COPY . .

RUN poetry install

EXPOSE 5000

CMD ["sh","start.sh"]
