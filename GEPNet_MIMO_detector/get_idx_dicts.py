def get_idx_dicts(user_num):
    #slicing parameters
    temp_a=[]
    temp_b=[]
    dict_index_val = {}
    dict_val_index = {}
    dict_final = {}
    
    for i in range(user_num):
        for j in range(user_num):
            if  i!=j:
                temp_a.append(i)
                temp_b.append(j)
                dict_final[len(temp_a)-1]=  str(j) + str(i)
                dict_index_val[len(temp_a)-1] = str(i) + str(j)
                dict_val_index[str(i) + str(j)] = len(temp_a)-1
                # dict_iv[t] = str(i) + str(j)
                # t+=user_num-1
                    
    return temp_a,temp_b
        