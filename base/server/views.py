from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("server/model/rf.pkl", "rb"))
ord_encoder = pickle.load(open("server/model/ord_encoder.pkl", "rb"))
ss  = pickle.load(open("server/model/ss.pkl", "rb"))

def IndexView(request):
    if request.method == "POST":
        columns = ['trust_apple', 'interest_computers', 'age_computer', 'user_pcmac',
       'appleproducts_count', 'familiarity_m1', 'f_batterylife', 'f_price',
       'f_size', 'f_multitasking', 'f_noise', 'f_performance', 'f_neural',
       'f_synergy', 'f_performanceloss', 'm1_consideration',
       'gender', 'age_group', 'income_group', 'status', 'domain']
        cat_cols = ['trust_apple', 'interest_computers', 'age_computer', 'user_pcmac',
                    'familiarity_m1', 'f_batterylife', 'f_price', 'f_multitasking', 'f_noise',
                    'f_performance', 'gender', 'status', 'domain']
        
        test_df = pd.DataFrame(columns=columns)
        test_df.loc[0, "trust_apple"] = request.POST["trust_apple"]
        test_df.loc[0, "interest_computers"] = request.POST["interest_computers"]
        test_df.loc[0, "age_computer"] = request.POST["age_computer"]
        test_df.loc[0, "user_pcmac"] = request.POST["user_pcmac"]
        test_df.loc[0, "appleproducts_count"] = request.POST["appleproducts_count"]
        test_df.loc[0, "familiarity_m1"] = request.POST["familiarity_m1"]
        test_df.loc[0, "f_batterylife"] = request.POST["f_batterylife"]
        test_df.loc[0, "f_price"] = request.POST["f_price"]
        test_df.loc[0, "f_size"] = request.POST["f_size"]
        test_df.loc[0, "f_multitasking"] = request.POST["f_multitasking"]
        test_df.loc[0, "f_noise"] = request.POST["f_noise"]
        test_df.loc[0, "f_performance"] = request.POST["f_performance"]
        test_df.loc[0, "f_neural"] = request.POST["f_neural"]
        test_df.loc[0, "f_synergy"] = request.POST["f_synergy"]
        test_df.loc[0, "f_performanceloss"] = request.POST["f_performanceloss"]
        test_df.loc[0, "m1_consideration"] = request.POST["m1_consideration"]
        test_df.loc[0, "gender"] = request.POST["gender"]
        test_df.loc[0, "age_group"] = request.POST["age_group"]
        test_df.loc[0, "income_group"] = request.POST["income_group"]
        test_df.loc[0, "status"] = request.POST["status"]
        test_df.loc[0, "domain"] = request.POST["domain"]
        #print(test_df)

        test_df[cat_cols] = ord_encoder.transform(test_df[cat_cols])
        test_df = ss.transform(test_df)
        result = model.predict(test_df)[0]
        result = "You need a M1 Mac." if result == "Yes" else "You dont need a M1 Mac."

        context = {"result": result}
        return render(request, "predict.html", context)

    return render(request, "index.html")