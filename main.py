from flask import render_template,Flask,request
app = Flask(__name__)
import pickle

file=open('model.pkl','rb')
cfl=pickle.load(file)

file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        my_dict=request.form
        fever=int(my_dict['temprature'])
        age = int(my_dict['age'])
        bodypain = int(my_dict['bodypain'])
        runnynose = int(my_dict['runnynose'])
        difficultbreathing=int(my_dict['difficultbreathing'])

        inputfeatures = [fever, bodypain, age, runnynose, difficultbreathing]
        print(inputfeatures)
        infect = cfl.predict([inputfeatures])
        noninf_prob=cfl.predict_proba([inputfeatures])[0][0]
        inf_prob = cfl.predict_proba([inputfeatures])[0][1]
        print(inf_prob)
        print(inf_prob)
        return render_template('show.html',infection=round(inf_prob*100),noninfection=round(noninf_prob*100))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
