# -*- encoding:utf8 -*-
#!/usr/bin/python3
entity2question= {
'疾病和诊断':'疾病和症状的描述',
'影像检查':'做的影像检查',
'实验室检验':'做的实验室细胞检验',
'手术':'手术名称',
'药物':'给予或服用药的名称',
'解剖部位':'所在部位',
}
entity_dic = {
        '疾病和诊断': 2,
        '影像检查': 4,
        '实验室检验': 6,
        '手术': 8,
        '药物': 10,
        '解剖部位': 12,
        'I-疾病和诊断': 3,
        'I-影像检查': 5,
        'I-实验室检验': 7,
        'I-手术': 9,
        'I-药物': 11,
        'I-解剖部位': 13
    }
    
  
def convert_to_column_format(filename):
    fieldir,name = os.path.split(filename)
    
    data = []
    for line in open(filename,'r',encoding='utf8'):
        data.append(json.loads(line))
    dataset = []
    for item in data:
        text = item['originalText']
        label = ['Others']*len(text) 
        entities = item['entities']
        
        for entity in entities:
            entity_start = entity['start_pos']
            entity_end = entity['end_pos']
            entity_type = entity['label_type']
            label[entity_start]= 'B-' + entity_type
            for ix in range(entity_start+1,entity_end):
                label[ix] = 'I-' + entity_type
        
        dataset.append([text,label])
    if 'train' in name:
        train_data = os.path.join(fieldir,'train_column.txt')
        valid_data = os.path.join(fieldir,'valid_column.txt')
        index_split = int(len(dataset)*0.8)
        dataset_train = dataset[:index_split]
        dataset_valid = dataset[index_split:]
        with open(train_data,'w',encoding= 'utf8') as f:
            for data in dataset_train:
                for ix in range(len(data[0])):
                    if data[0][ix] != '。':
                        f.write( data[0][ix]+ ' '+  data[1][ix])
                    f.write('\n')
        with open(valid_data,'w',encoding= 'utf8') as f:
            for data in dataset_valid:
                for ix in range(len(data[0])):
                    if data[0][ix] != '。':
                        f.write( data[0][ix]+ ' '+  data[1][ix])
                    f.write('\n')
    else:
        test_data = os.path.join(fieldir,'test_column.txt')
        with open(test_data,'w',encoding= 'utf8') as f:
            for data in dataset:
                for ix in range(len(data[0])):
                    if data[0][ix] != '。':
                        f.write( data[0][ix]+ ' '+  data[1][ix])
                    f.write('\n')


def parse_line(text_list, ann_list,entity_count_dic):
    text = ''.join(text_list)

    ix = 0
    ann_len = len(ann_list)
    ans_dic = {}
    data_list = []
    while ix < ann_len :
        if 'B-' in ann_list[ix]:
            entity = ann_list[ix].split('-')[-1]
            if entity not in entity_count_dic:
                entity_count_dic[entity] = 1
            else:
                entity_count_dic[entity] += 1
            if entity not in ans_dic:
                ans_dic[entity]= []
            start_index = ix
            ix += 1
            while ix < ann_len and ann_list[ix]!='Others' and  entity ==  ann_list[ix].split('-')[-1]:
                ix += 1
            end_index = ix

            ans_dic[entity].append([start_index,end_index,entity])
            ix -= 1
        ix+=1
    for entity in entity2question:
        if entity in ans_dic:
            data_list.append([text,entity2question[entity],ans_dic[entity]])


    return  data_list
def adjust_index(text_list,ann_list):
    assert len(text_list) == len(ann_list)
    index = len(text_list) - 1
    while index>=0 and ann_list[index]!='Others':
        index -= 1
    return index

def rectify_line(text_list,ann_list,entity_rectified_count_dic):
    index = 0
    rectify_count = 0
    len_data = len(text_list)
    text_rectified = []
    ann_rectified = []

    while index < len_data:
        line = text_list[index]
        if ' '== line and  ann_list[index]!='Others':
            text_rectified.append(' ')
            ann_rectified.append('Others')
            index += 1
            entity = ann_list[index].split('-')[-1]
            if entity not in entity_rectified_count_dic:
                
                entity_rectified_count_dic[entity] = 1
            else:
                entity_rectified_count_dic[entity] += 1
            rectify_count += 1
            while index < len_data and text_list[index] == ' ':
                text_rectified.append(' ')
                ann_rectified.append('Others')
                index += 1
            if index < len_data:
               
                text_rectified.append(text_list[index])

                if 'I-' in ann_list[index]:
                    ann_rectified.append('B-' + ann_list[index].split('-')[-1])
                else:
                    ann_rectified.append(ann_list[index]) 
                    
                index += 1
                if index < len_data and text_list[index-1] in ['胃','腹','肠','子宫','胸'] and  text_list[index] in ['壁','腔','管'] and  ann_list[index] == 'Others' and  ann_list[index-1] != 'Others' :
                    entity = ann_list[index-1].split('-')[-1]
                    text_rectified.append(text_list[index])
                    ann_rectified.append('I-'+ entity)
                    index += 1
                    
                
        else:
            text_rectified.append(text_list[index])
            ann_rectified.append(ann_list[index])
            index += 1

            if index < len_data and text_list[index-1] in ['胃','腹','肠','子宫','胸'] and  text_list[index] in ['壁','腔','管'] and  ann_list[index] == 'Others'and  ann_list[index-1] != 'Others'  :
                entity = ann_list[index-1].split('-')[-1]
                text_rectified.append(text_list[index])
                ann_rectified.append('I-'+ entity)
                index += 1
    
    return text_rectified,ann_rectified,rectify_count
    
def line_split(text_list,ann_list,max_length):
    res = []
    start_index = 0
    end_index = max_length
    line_length = len(text_list)
    if line_length < max_length:
        return [text_list,ann_list]
    else:
        while end_index < line_length:
            
            if ann_list[end_index] !='Others':
                end_index = adjust_index(text_list[start_index:end_index],ann_list[start_index:end_index])
            
            res.append([text_list[start_index:end_index],ann_list[start_index:end_index]])
            start_index = end_index
            end_index += max_length
        if start_index <=  line_length and end_index > line_length:
             res.append([text_list[start_index:],ann_list[start_index:]])
    return res
    with open(train_file,'r',encoding='utf8')as f:
        data  = f.readlines()
    ix = 0
    len_text = 0 
    len_data = len(data)
    entity_count_dic = {}
    text_list = []
    ann_list = []
    dataset = []
    dataset_long = []
    dataset_short = []
    while ix < len_data:
        while ix < len_data and data[ix]!='\n':
            line = re.sub(r'\n','',data[ix]).split(' ')
            text = line[0] if line[0]!='' else ' '
            ann = line[-1]
           
            text_list.append(text)
            ann_list.append(ann)
            ix+=1
        len_text = max(len_text,len(text_list))
        if len(text_list) > 256:

            line_split_res = line_split(text_list,ann_list,max_length)
            for item in line_split_res:
                data_line =parse_line(item[0],item[1],entity_count_dic)
                dataset_short.append([item[0],item[1]])
                dataset.extend(data_line)
            
        else:
            dataset_short.append([text_list,ann_list])
            data_line =parse_line(text_list,ann_list,entity_count_dic)
            dataset.extend(data_line)
        text_list = []
        ann_list = []
        ix += 1
    print('max_seq_length = ',len_text)
    print('entity_count_dic= ',entity_count_dic)
    print('data_split =',len(dataset_short))
    out_name = train_file.split('.')[0]+'_qa.json'
    with open(out_name,'w',encoding='utf8') as f:
        json.dump(dataset,f,indent=4,ensure_ascii=False)

            
    with open(train_file.split('.')[0]+'_short.txt','w',encoding='utf8') as f:
        for data in dataset_short:
            for ix in range(len(data[0])):
                f.write(data[0][ix]+' '+data[1][ix]+'\n')
            f.write('\n')
    dataset_short_split = dataset_short[:int(len(dataset_short)*0.75)]
    with open(train_file.split('.')[0]+'_short_split75.txt','w',encoding='utf8') as f:
        for data in dataset_short_split:
            for ix in range(len(data[0])):
                f.write(data[0][ix]+' '+data[1][ix]+'\n')
            f.write('\n')

def convert_flair_to_IE(file_dir):
    train_file = os.path.join(file_dir, 'train_short.txt')
    valid_file = os.path.join(file_dir, 'valid_short.txt')
    test_file = os.path.join(file_dir, 'test_short.txt')
    print('train')
    cut_sentences(train_file,max_length=256)
    print('valid')
    cut_sentences(valid_file,max_length=256)
    print('test')
    cut_sentences(test_file,max_length=256)
            

    
def convert_to_context_question_answer(train_file):
    with open(train_file,'r',encoding='utf8') as  f:
        dataset = []
        ans = []

        data_dic = json.load(f)
        for context, qas in data_dic.items():
            context = re.sub(',', '，', context)
            entity_dic = {}
            for question,answer in qas.items():
                #find the entity token
                if type(answer[0]) ==list:
                    entity_dic[answer[0][2]] = context[answer[0][0]:answer[0][1]]
                else:
                    entity_dic[answer[2]] = context[answer[0]:answer[1]]

                question = re.sub(',', '，', question)
                dataset.append([context+'。',
                               question+'？',
                               answer])

        return dataset

def cut_sentences(train_file,max_length=256):
    with open(train_file,'r',encoding='utf8')as f:
        data  = f.readlines()
    ix = 0
    len_text = 0 
    len_data = len(data)
    entity_rectified_count_dic = {}
    text_list = []
    ann_list = []
    dataset = []
    dataset_long = []
    dataset_short = []
    rectify_count = 0
    while ix < len_data:
        while ix < len_data and data[ix]!='\n':
            line = re.sub(r'\n','',data[ix]).split(' ')
            text = line[0] if line[0]!='' else ' '
            ann = line[-1]
           
            text_list.append(text)
            ann_list.append(ann)
            ix+=1
        len_text = max(len_text,len(text_list))
        if len(text_list) > 256:

            line_split_res = line_split(text_list,ann_list,max_length)
            for item in line_split_res:
                text_rectified,ann_rectified,rec_count =  rectify_line(item[0],item[1],entity_rectified_count_dic)
                dataset_short.append([text_rectified,ann_rectified])
                rectify_count += rec_count
            
        else:
            text_rectified,ann_rectified,rec_count = rectify_line(text_list,ann_list,entity_rectified_count_dic)
            dataset_short.append([text_rectified,ann_rectified])
        text_list = []
        ann_list = []
        ix += 1
    print('max_seq_length = ',len_text)
    print('data_split =',len(dataset_short))
    print('entity_rectified_count_dic =',entity_rectified_count_dic)

    with open(train_file.split('.')[0]+'_short_rectified.txt','w',encoding='utf8') as f:
        for data in dataset_short:
            for ix in range(len(data[0])):
                try:
                    f.write(data[0][ix]+' '+data[1][ix]+'\n')
                except Exception  as e :
                    print('exception  = {},data ={} '.format(data,e))
                    
            f.write('\n')
    dataset_short_split = dataset_short[:int(len(dataset_short)*0.75)]
    with open(train_file.split('.')[0]+'_short_rectified_split75.txt','w',encoding='utf8') as f:
        for data in dataset_short_split:
            for ix in range(len(data[0])):
                f.write(data[0][ix]+' '+data[1][ix]+'\n')
            f.write('\n')

def convert_to_short_sent(file_dir):
    train_file = os.path.join(file_dir, 'train_column.txt')
    valid_file = os.path.join(file_dir, 'valid_column.txt')
    test_file = os.path.join(file_dir, 'test_column.txt')
    print('train')
    cut_sentences(train_file)
    print('valid')
    cut_sentences(valid_file)
    print('test')
    cut_sentences(test_file)

    return features,context_question_tokens
if __name__ == "__main__":
    convert_to_column_format('./test_data.json')
    convert_to_column_format('./train_data.json')
    convert_to_short_sent('./')