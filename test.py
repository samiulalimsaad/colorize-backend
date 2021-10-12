import base64


imgList = []
with open(f"obama.jpg", "rb") as image_file:
    data = base64.b64encode(image_file.read())
    imgList.append(b"data:image/jpeg;base64,"+data)

print(type(str(imgList[0])))