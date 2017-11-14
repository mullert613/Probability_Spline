import generate_splines
from generate_splines import generate_splines_fun

if __name__ == '__main__':
	remove_index = [0,1,2,3,4,5,6]
	for x in remove_index:
		generate_splines.generate_splines_fun(remove_index=x)




