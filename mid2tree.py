import sys

def tree(acts):
	btree = []
	openidx = []
	wid = 0
	for act in acts:
		if act[0] == 'S':
			tmp = act.split()
			btree.append("("+tmp[1]+" "+tmp[2]+")")
			wid += 1
		#elif act[0] == 'N':
		#	btree.insert(-1,"("+act[3:-1])
		#	openidx.append(len(btree)-2)
		else:
			nt=act[3:-1]
			a=nt.split('#', 1 )
			#print nt, a[0], a[1]
			#openidx.append(int(a[1])-2)
			btree.insert(-1*int(a[1]),"("+a[0])
			#print "1", btree
			tmp = " ".join(btree[len(btree)-int(a[1])-1:])+")"
			btree = btree[:len(btree)-int(a[1])-1]
			#print "2", btree
			btree.append(tmp)
			#print "3", btree
			#openidx = openidx[:-1]
	print btree[0]

if __name__ == "__main__":
	actions = []
	action = []	
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":
			actions.append(action[:-1])
			action = []
		else:
			action.append(line)
	#print actions
	for i in range(len(actions)):	
		tree(actions[i]);
