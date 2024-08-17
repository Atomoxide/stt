import logging
import re
from flask import Flask, request, jsonify
import os
from gevent.pywsgi import WSGIServer, WSGIHandler
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')
from stslib import cfg
from stslib.cfg import ROOT_DIR
from faster_whisper import WhisperModel
import glob

class CustomRequestHandler(WSGIHandler):
    def log_request(self):
        pass


# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)
app = Flask(__name__, static_folder=os.path.join(ROOT_DIR, 'static'), static_url_path='/static',  template_folder=os.path.join(ROOT_DIR, 'templates'))
root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)

# 配置日志
app.logger.setLevel(logging.WARNING)  # 设置日志级别为 INFO
# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
file_handler = RotatingFileHandler(os.path.join(ROOT_DIR, 'sts.log'), maxBytes=1024 * 1024, backupCount=5)
# 创建日志的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器的级别和格式
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
app.logger.addHandler(file_handler)


@app.route('/api',methods=['GET','POST'])
def api():
    try:
        # 获取上传的文件
        audio_file = request.files['file']
        model = request.form.get("model")
        language = request.form.get("language")
        response_format = "json"
        print(f'{model=},{language=},{response_format=}')
        if not os.path.exists(os.path.join(cfg.MODEL_DIR, f'models--Systran--faster-whisper-{model}/snapshots/')):
            return jsonify({"code": 1, "msg": f"{model} {cfg.transobj['lang4']}"})

        noextname, ext = os.path.splitext(audio_file.filename)
        ext = ext.lower()
        # save .wav to temo
        wav_file = os.path.join(cfg.TMP_DIR, f'{noextname}.wav')
        audio_file.save(wav_file)
        sets=cfg.parse_ini()
        model = WhisperModel(model, device=sets.get('devtype'), compute_type=sets.get('cuda_com_type'), download_root=cfg.ROOT_DIR + "/models", local_files_only=True)

        # invoke Faster Whisper
        segments,_ = model.transcribe(wav_file, beam_size=sets.get('beam_size'),best_of=sets.get('best_of'),temperature=0 if sets.get('temperature')==0 else [0.0,0.2,0.4,0.6,0.8,1.0],condition_on_previous_text=sets.get('condition_on_previous_text'),vad_filter=sets.get('vad'),
    vad_parameters=dict(min_silence_duration_ms=300,max_speech_duration_s=10.5),language=language,initial_prompt=None if language!='zh' else sets.get('initial_prompt_zh'))
        os.remove(wav_file)
        # clauses concat
        clauses =[]
        for  segment in segments:
            clause = segment.text.strip().replace('&#39;', "'")
            clause = re.sub(r'&#\d+;', '', clause)

            # 无有效字符
            if not clause or re.match(r'^[，。、？‘’“”；：（｛｝【】）:;"\'\s \d`!@#$%^&*()_+=.,?/\\-]*$', clause) or len(clause) <= 1:
                continue
            clauses.append(clause)
        sentence = " ".join(clauses)
        return jsonify({"code": 0, "msg": 'ok', "data": sentence})
    except Exception as e:
        print(e)
        app.logger.error(f'[api]error: {e}')
        return jsonify({'code': 2, 'msg': str(e)})



if __name__ == '__main__':
    http_server = None
    try:
        try:
            if cfg.devtype=='cpu':
                print('\n如果设备使用英伟达显卡并且CUDA环境已正确安装，可修改set.ini中\ndevtype=cpu 为 devtype=cuda, 然后重新启动以加快识别速度\n')
            host = cfg.web_address.split(':')
            http_server = WSGIServer((host[0], int(host[1])), app, handler_class=CustomRequestHandler)
            http_server.serve_forever()
        finally:
            if http_server:
                http_server.stop()
    except KeyboardInterrupt:
        print("Keyborad Stop")
        http_server.stop()
        files_to_purge = glob.glob("./static/tmp/*")
        for file in files_to_purge:
            os.remove(file)
    except Exception as e:
        if http_server:
            http_server.stop()
        print("error:" + str(e))
        app.logger.error(f"[app]start error:{str(e)}")
