from client_api import configure,get_auth_key,shutdown_server,get_label_names,create_prediction_folder,log_out,create_user,add_image_to_prediction,make_mass_prediction


configure("localhost")

key=get_auth_key('admin','papuga12')
print(key)
create_user(key,'dawid','haslo')
log_out(key)

key=get_auth_key('dawid','haslo')
print(key)
print(get_label_names(key))
folder=create_prediction_folder(key)
add_image_to_prediction(key,folder,'1.jpeg')
add_image_to_prediction(key,folder,'2.jpeg')
add_image_to_prediction(key,folder,'3.jpeg')
print(make_mass_prediction(key,folder))


log_out(key)



#key=get_auth_key('admin','papuga12')
#print(key)
#shutdown_server(key)