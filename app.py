
import base64

from flask import Flask, request
from flask_cors import CORS

from main import execute

app = Flask(__name__)
CORS(app)


@app.route('/',methods=["GET"])
def get():
    imgList = []
    with open(f"obama.jpg", "rb") as image_file:
        data = base64.b64encode(image_file.read())
        imgList.append(b"data:image/jpeg;base64,"+data)

    data = imgList[0]
    print(data)
    print(type(data))
    return data

@app.route('/', methods=["POST"])
def post():
    body = request.get_json()
    image = body.get("image")
    # return {'error': "only image attribute allow"}
    if image is not None:
        try:
            res = execute(image=image)
            return res
        except Exception as e:
            return None
    else:
        return {'error': "only image attribute allow"}
        


if __name__ == '__main__':
    app.run(debug=True)
