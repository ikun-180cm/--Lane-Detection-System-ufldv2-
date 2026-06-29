import os
import os.path
from flask import Flask, render_template, request, redirect, session, flash,url_for
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from ufdv2.inference import run_ufld_on_image, run_ufld_on_video

app = Flask(__name__)
app.secret_key = "123456"

# ===================== 固定配置 =====================
ADMIN_KEY = "$1743552$"
UPLOAD_FOLDER = "static/uploads"
AVATAR_FOLDER = "static/avatar"
RESULT_FOLDER = "static/results"
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['AVATAR_FOLDER'] = AVATAR_FOLDER
app.config['RESULT_FOLDER'] = os.path.join(os.getcwd(), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm'}
ALLOWED_AVATAR = {'png', 'jpg', 'jpeg'}

os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AVATAR_FOLDER, exist_ok=True)

def get_db():
    return sqlite3.connect("车道检测数据库.db")

# ===================== 数据库升级（保留数据） =====================
def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user',
        avatar TEXT DEFAULT '',
        gender TEXT DEFAULT '未知'
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS record (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filename TEXT,
        upload_time TEXT
    )""")
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_avatar(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AVATAR


def delete_record_and_files(record_id, username=None, is_admin=False):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # 1. 查询要删除的文件名称
        if is_admin:
            cursor.execute("SELECT filename, username FROM record WHERE id=?", (record_id,))
            res = cursor.fetchone()
            if res:
                filename, record_username = res[0], res[1]
        else:
            cursor.execute("SELECT filename FROM record WHERE id=? AND username=?", (record_id, username))
            res = cursor.fetchone()
            if res:
                filename = res[0]

        # 2. 删除本地文件（上传文件+结果文件）
        if res:
            # 上传文件路径
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            # 结果文件路径
            base, ext = os.path.splitext(filename)
            result_filename = f"{base}_result{ext}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            if os.path.exists(result_path):
                os.remove(result_path)

        # 3. 删除数据库记录
        if is_admin:
            cursor.execute("DELETE FROM record WHERE id=?", (record_id,))
        else:
            cursor.execute("DELETE FROM record WHERE id=? AND username=?", (record_id, username))

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"删除失败：{e}")
        return False
    finally:
        conn.close()

def batch_delete_records(ids, username=None, is_admin=False):
    for record_id in ids:
        delete_record_and_files(record_id, username=username, is_admin=is_admin)

# ===================== 首页 =====================
@app.route('/')
def index():
    if 'username' not in session:
        return redirect('/login')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT avatar, gender FROM user WHERE username=?", (session['username'],))
    user = cursor.fetchone()
    conn.close()
    if not user:
        session.clear()
        return render_template("error.html", message="账号已被管理员删除", btn_text="重新登录", back_url="/login")
    gender_symbol = "?"
    if user[1] == "男": gender_symbol = "♂"
    if user[1] == "女": gender_symbol = "♀"
    return render_template("index.html", avatar=user[0], gender=gender_symbol)

# ===================== 登录 =====================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'user')
        admin_key = request.form.get('admin_key', '')
        if role == 'admin' and admin_key != ADMIN_KEY:
            return render_template("error.html", message="管理员密钥错误", btn_text="返回登录", back_url="/login")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user WHERE username=? AND password=? AND role=?",
                      (username, password, role))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            session['role'] = role
            return redirect('/')
        else:
            return render_template("error.html", message="用户名/密码/身份错误", btn_text="返回登录", back_url="/login")
    return render_template("login.html")

# ===================== 注册 =====================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role','user')
        admin_key = request.form.get('admin_key','')
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user WHERE username=?", (username,))
        if cursor.fetchone():
            conn.close()
            return render_template("error.html", message="用户名已存在", btn_text="返回注册", back_url="/register")
        if role == 'admin' and admin_key != ADMIN_KEY:
            conn.close()
            return render_template("error.html", message="管理员密钥错误", btn_text="返回注册", back_url="/register")
        cursor.execute("INSERT INTO user (username,password,role) VALUES (?,?,?)",
                      (username, password, role))
        conn.commit()
        conn.close()
        return redirect('/login')
    return render_template("register.html")

# ===================== 退出 =====================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ===================== 上传页面（GET） =====================
@app.route('/upload')
def upload_page():
    if 'username' not in session: return redirect('/login')
    return render_template("upload.html")

# ===================== 上传（POST） =====================
@app.route('/upload', methods=['POST'])
def upload():
    if 'username' not in session: return redirect('/login')
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template("error.html", message="未选择文件", btn_text="返回首页", back_url="/")
    if not allowed_file(file.filename):
        return render_template("error.html", message="不支持的文件格式", btn_text="返回首页", back_url="/")
    filename = secure_filename(file.filename)
    base, ext = os.path.splitext(filename)
    new_filename = filename
    counter = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    try:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
        input_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            new_filename
        )
        base, ext = os.path.splitext(new_filename)
        result_filename = f"{base}_result{ext}"
        result_path = os.path.join(
            app.config['RESULT_FOLDER'],
            result_filename
        )
        video_exts = {
            '.mp4',
            '.avi',
            '.mov',
            '.mkv',
            '.flv',
            '.wmv',
            '.webm'
        }
        image_exts = {
            '.jpg',
            '.jpeg',
            '.png'
        }
        ext = ext.lower()
        if ext in video_exts:
            run_ufld_on_video(
                input_path,
                result_path
            )
        elif ext in image_exts:
            result_img = run_ufld_on_image(
                input_path
            )
            import cv2
            cv2.imwrite(
                result_path,
                result_img
            )
        else:
            raise ValueError("不支持的文件格式")
        conn = get_db()
        cursor = conn.cursor()
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO record (username,filename,upload_time) VALUES (?,?,?)",
                      (session['username'], new_filename, upload_time))
        conn.commit()
    except Exception as e:
        return render_template("error.html", message=f"上传失败：{str(e)}", btn_text="返回首页", back_url="/")
    finally:
        if 'conn' in locals(): conn.close()
    # return redirect(f'/result?filename={result_filename}&original={new_filename}')
    return redirect(url_for('result', filename=result_filename, original=new_filename))

# ===================== 上传结果 =====================
@app.route('/result')
def result():
    if 'username' not in session:
        return redirect('/login')
    filename = request.args.get('filename')
    original_filename = request.args.get('original', filename)

    if not filename:
        return "参数错误", 400

    # 生成文件URL
    result_url = url_for('static', filename=f'results/{filename}')
    original_url = url_for('static', filename=f'uploads/{original_filename}')

    # 补充文件类型判断
    def get_file_type(fn):
        ext = os.path.splitext(fn)[1].lower().lstrip('.')
        if ext in {'jpg', 'jpeg', 'png'}:
            return 'image'
        elif ext in {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}:
            return 'video'
        return 'other'

    original_type = get_file_type(original_filename)
    result_type = get_file_type(filename)

    # 传给前端页面
    return render_template(
        "result.html",
        result_url=result_url,
        original_url=original_url,
        filename=filename,
        original_filename=original_filename,
        original_type=original_type,
        result_type=result_type
    )


# ===================== 历史记录 =====================
@app.route('/history')
def history():
    if 'username' not in session: return redirect('/login')
    page = request.args.get('page',1,int)
    per_page=5
    offset=(page-1)*per_page
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM record WHERE username=?",(session['username'],))
    total=cursor.fetchone()[0]
    total_pages=(total+per_page-1)//per_page
    cursor.execute("SELECT id,username,filename,upload_time FROM record WHERE username=? ORDER BY id DESC LIMIT ? OFFSET ?",
                  (session['username'],per_page,offset))
    records=cursor.fetchall()
    conn.close()
    return render_template("history.html", records=records, page=page, total_pages=total_pages)

# ===================== 头像上传 =====================
@app.route('/avatar',methods=['GET','POST'])
def avatar():
    if 'username' not in session: return redirect('/login')
    if request.method=='POST':
        file=request.files.get('avatar')
        if file and allowed_avatar(file.filename):
            _,ext=os.path.splitext(file.filename)
            filename=f"{session['username']}{ext}"
            file.save(os.path.join(AVATAR_FOLDER,filename))
            conn=get_db()
            cursor=conn.cursor()
            cursor.execute("UPDATE user SET avatar=? WHERE username=?",(filename,session['username']))
            conn.commit()
            conn.close()
            flash("头像上传成功！","success")
            return redirect('/')
    return render_template("avatar.html")

# ===================== 性别修改 =====================
@app.route('/gender',methods=['POST'])
def gender():
    if 'username' not in session: return redirect('/login')
    g=request.form.get('gender')
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("UPDATE user SET gender=? WHERE username=?",(g,session['username']))
    conn.commit()
    conn.close()
    flash("性别修改成功！","success")
    return redirect('/')

# ===================== 忘记密码 =====================
@app.route('/forget',methods=['GET','POST'])
def forget():
    if request.method=='POST':
        username=request.form['username']
        new_pwd=request.form['password']
        conn=get_db()
        cursor=conn.cursor()
        cursor.execute("SELECT * FROM user WHERE username=?",(username,))
        if not cursor.fetchone():
            conn.close()
            flash("用户不存在","error")
            return redirect('/forget')
        cursor.execute("UPDATE user SET password=? WHERE username=?",(new_pwd,username))
        conn.commit()
        conn.close()
        flash("密码重置成功，请登录！","success")
        return redirect('/login')
    return render_template("forget.html")

# ===================== 管理员：用户列表 =====================
@app.route('/admin/users')
def admin_users():
    if session.get('role')!='admin': return redirect('/login')
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("SELECT id,username,role,gender,avatar FROM user")
    users=cursor.fetchall()
    conn.close()
    return render_template("admin_users.html",users=users)

# ===================== 管理员：改密 =====================
@app.route('/admin/reset_pwd',methods=['POST'])
def admin_reset_pwd():
    if session.get('role')!='admin': return redirect('/login')
    uid=request.form.get('id')
    new_pwd=request.form.get('password')
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("UPDATE user SET password=? WHERE id=?",(new_pwd,uid))
    conn.commit()
    conn.close()
    flash(f"密码修改成功！新密码：{new_pwd}","success")
    return redirect('/admin/users')

# ===================== 管理员：删用户 =====================
@app.route('/admin/del_user', methods=['POST'])
def admin_del_user():
    if session.get('role') != 'admin':
        return redirect('/login')
    uid = request.form.get('id')
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT username FROM user WHERE id=?", (uid,))
    username = cursor.fetchone()[0]

    cursor.execute("SELECT filename FROM record WHERE username=?", (username,))
    files = cursor.fetchall()
    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f[0])
        if os.path.exists(filepath):
            os.remove(filepath)

        base, ext = os.path.splitext(f[0])
        result_filename = f"{base}_result{ext}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        if os.path.exists(result_path):
            os.remove(result_path)

    cursor.execute("DELETE FROM record WHERE username=?", (username,))
    cursor.execute("DELETE FROM user WHERE id=?", (uid,))
    conn.commit()
    conn.close()

    flash("用户及所有记录删除成功！", "success")
    return redirect('/admin/users')

# ===================== 管理员：查看记录 =====================
@app.route('/admin/history/<username>')
def admin_history(username):
    if session.get('role')!='admin': return redirect('/login')
    page=request.args.get('page',1,int)
    per_page=5
    offset=(page-1)*per_page
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM record WHERE username=?",(username,))
    total=cursor.fetchone()[0]
    total_pages=(total+per_page-1)//per_page
    cursor.execute("SELECT id,username,filename,upload_time FROM record WHERE username=? ORDER BY id DESC LIMIT ? OFFSET ?",
                  (username,per_page,offset))
    records=cursor.fetchall()
    conn.close()
    return render_template("admin_history.html",records=records,username=username,page=page,total_pages=total_pages)


@app.route('/user/del_record', methods=['POST'])
def user_del_record():
    if 'username' not in session:
        return redirect('/login')

    record_id = request.form.get('id')
    if not record_id:
        flash("参数错误，删除失败", "error")
        return redirect('/history')
    username = session['username']

    # 调用通用删除函数
    delete_record_and_files(record_id, username, is_admin=False)

    flash("删除成功！", "success")
    return redirect('/history')


@app.route('/user/batch_del', methods=['POST'])
def user_batch_del():
    if 'username' not in session:
        return redirect('/login')

    ids = request.form.getlist('record_ids')
    if not ids:
        flash("请选择要删除的记录！", "error")
        return redirect('/history')

    # 调用通用批量删除
    batch_delete_records(ids, username=session['username'], is_admin=False)

    flash("批量删除成功！", "success")
    return redirect('/history')

# ===================== 管理员：删单条记录 =====================
@app.route('/admin/del_record', methods=['POST'])
def admin_del_record():
    if session.get('role') != 'admin':
        return redirect('/login')

    record_id = request.form.get('id')
    username = request.form.get('username')
    if not record_id or not username:
        flash("参数错误，删除失败", "error")
        return redirect(f'/admin/history/{username}')

    # 调用通用函数（管理员模式）
    delete_record_and_files(record_id, is_admin=True)

    flash("删除成功！", "success")
    return redirect(f'/admin/history/{username}')

# ===================== 管理员：批量删除 =====================
@app.route('/admin/batch_del', methods=['POST'])
def admin_batch_del():
    if session.get('role') != 'admin':
        return redirect('/login')

    ids = request.form.getlist('record_ids')
    username = request.form.get('username')

    if not ids:
        flash("请选择要删除的记录！", "error")
        return redirect(f'/admin/history/{username}')

    # 调用通用批量删除（管理员模式）
    batch_delete_records(ids, is_admin=True)

    flash("批量删除成功！", "success")
    return redirect(f'/admin/history/{username}')

if __name__ == "__main__":
    init_db()
    app.run(debug=True)