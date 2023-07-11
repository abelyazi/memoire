


def check_max_pred(prd):
  max=0
  for i in prd:
    if i[0] > max:
      max = i[0]
  print(max)

def cumulative(lists): 
    cu_list = [] 
    length = len(lists) 
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)] 
    return cu_list[1:]

def mean_pred(prediction):
  pred = []
  for i in prediction:
    lst = [ j[0] for j in i]
    avg = sum(lst)/len(lst)
    if avg > 0.5:
      pred.append(1)
    else:
      pred.append(0)
  return pred

def majority_pred(prediction):
  pred = []
  for i in prediction:
    lst = [ j[0] for j in i]
    if (sum(elem > 0.5 for elem in lst)) > (len(lst)/2):
      pred.append(1)
    else:
      pred.append(0)
  return pred
