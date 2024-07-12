from flask import Flask, request, jsonify
from capture import capture_function  # 导入capture.py中的函数
from readfile import readfile_function  # 导入readfile.py中的函数

app = Flask(__name__)

@app.route('/api/capture', methods=['POST'])
def api_capture():
    data = request.json.get('data')
    result = capture_function(data)
    return jsonify({'result': result})

@app.route('/api/readfile', methods=['POST'])
def api_readfile():
    data = request.json.get('data')
    result = readfile_function(data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # 使用host='0.0.0.0'使其在所有网络接口上运行
