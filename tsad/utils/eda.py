import pandas as pd 

def value_counts_interval(array,itervals):
    """
    input : np.array, list of values
    retrun : pd.series
    """
    names = [f"до {itervals[0]}"]
    quantity = [len(array[array < itervals[0]])]
    for i in range(len(itervals)-1):
        quantity.append(len(array[(array >= itervals[i]) & (array < itervals[i+1])]))
        names.append(f'c {itervals[i]} до {itervals[i+1]}')
    names += [f"от {itervals[-1]}"]
    quantity += [len(array[array >= itervals[-1]])]
    ts = pd.Series(quantity,index=names)    
    return ts



# def (df,num_points):
#     """ 
#     Посмотреть среднее расстояние между всеми парами точек (сэмлов) 
#     в первых num_points точках
#     в последних num_points точках
#     и одновременно в первых и последжних num_points точках.
#     """
#     import itertools
#     import scipy 
#     array1 = df.iloc[:num_points].values
#     array2 = df.iloc[num_points:int(2*num_points)].values

#     indexes = list(range(len(array1)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = array1[np.array(pairs)[:,0]]
#     list2 = array1[np.array(pairs)[:,1]]
#     print('Claster 1', scipy.spatial.distance.cdist(list1,list2).mean())


#     indexes = list(range(len(array2)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = array2[np.array(pairs)[:,0]]
#     list2 = array2[np.array(pairs)[:,1]]
#     print('Claster 2', scipy.spatial.distance.cdist(list1,list2).mean())

#     common_array = np.concatenate([array1,array2])
#     indexes = list(range(len(common_array)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = common_array[np.array(pairs)[:,0]]
#     list2 = common_array[np.array(pairs)[:,1]]
#     print('Claster common', scipy.spatial.distance.cdist(list1,list2).mean())
