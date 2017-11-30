import numpy as np
import pickle


alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' "
char_dict = {}
idx = 1

for char in alphabets:
	char_dict[char] = idx
	idx += 1

with open('sloAnv.pkl','rb') as f:
	sloAnv = pickle.load(f)
# sloAnv = pickle.load(open('sloAnv.pkl', 'rb'))

encoded_slo_data = []
encoded_anv_data = []
for i in range(len(sloAnv)):
	try:
		encoded_slo_line = []
		for item in sloAnv[i]['slo']:
			for char in item:
				encoded_slo_line.append(char_dict[char])
			encoded_slo_line.append(char_dict[' '])

		encoded_anv_line = []
		for item in sloAnv[i]['anv']:
			for char in item:
				encoded_anv_line.append(char_dict[char])
			encoded_anv_line.append(char_dict[' '])

		encoded_slo_data.append(encoded_slo_line)
		encoded_anv_data.append(encoded_anv_line)
	except:
		continue

pickle.dump(encoded_slo_data, open('encoded_slo_data.pkl','wb'))
pickle.dump(encoded_anv_data, open('encoded_anv_data.pkl','wb'))
