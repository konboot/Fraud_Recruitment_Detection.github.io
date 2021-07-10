from flask import Flask,render_template, url_for, redirect, request
import csv
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/connect.php', methods =['GET', 'POST'])
def submit():
    if request.method == "POST":
         try: 
             data = request.form.to_dict()
             write_data_csv = data
             return render_template("thank you.html")
         except:
             return 'Did not submit data into database'
    else:
         return "Error in submitting the form. Please try again!"
@app.route('/<string:page_name>')
def page(page_name='/'):
    try:
        return render_template(page_name)
    except:
        return redirect('/')

def write_data_csv(data):
    Email = data['Email']
    Subject = data['Subject']
    Message = data['Message']
    with open('db.csv', 'w', newline='') as csvfile:
       db_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
       db_writer.writerow([Email, Subject, Message])