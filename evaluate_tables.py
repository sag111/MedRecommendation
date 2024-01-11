import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from logging import warnings
import argparse
import os
import json

columns_metric_types = {
    "drugs": "strings_set_comparison",
    "drugs_top5": "strings_set_comparison",
    "drugs_best": "strings_set_comparison",
    "ADR_all": "strings_set_comparison",
    "ADR_top5": "strings_set_comparison",
    "Quality": "classification_f1_score"
}

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='evaluate_tables',
                    description='Скрипт для сравнения двух таблиц с данными собранными по коллекции отзывов')
    parser.add_argument('-t', '--true', help="файл с эталонными данными", required=True)
    parser.add_argument('-p', '--pred', help="файл с разультатами предсказания", required=True)
    parser.add_argument('--main_col', help="название основной колонки в файле, которая служит входными данными", required=True)
    parser.add_argument('--target_cols', help="названия колонок для которых нужно посчитать оценки", nargs="+", required=True)
    parser.add_argument('--output', help="путь к файлу, куда сохранить оценки", default=None)
    args = parser.parse_args()

    if args.output is not None and os.path.exists(args.output):
        warnings.warn(f"Файл для сохранения оценок уже существует и будет перезаписан: {args.output}.")

    gold = pd.read_csv(args.true)
    predict = pd.read_csv(args.pred)
    main_column = args.main_col
    target_columns = args.target_cols

    report_json = {}
    # проверим, что указанные колонки (основная и те, по которым будут считаться оценки) действительно есть в указанных файлах
    if not set([main_column] + target_columns).issubset(gold.columns):
        raise ValueError("Некоторые из колонок {} отсутствуют в таблице {} с колонками: {}".format(set([main_column] + target_columns), args.true, gold.columns))
    if not set([main_column] + target_columns).issubset(predict.columns):
        raise ValueError("Некоторые из колонок {} отсутствуют в таблице {} с колонками: {}".format(set([main_column] + target_columns), args.pred, predict.columns))

    gold_inputs = gold[main_column].values
    pred_inputs = predict[main_column].values
    # проверим, что основной колонке нет дубликатов. Такого в задаче не должно быть и скорее всего это указывает на ошибку при составлении таблиц
    assert len(gold_inputs)==len(set(gold_inputs))
    if len(pred_inputs)!=len(set(pred_inputs)):
        warnings.warn(f"Количество значений в основной колонке '{main_column}' не равно количеству уникальных значений в ней. Вероятно присутствую дубликаты. Не должно быть. Будет учитываться только первый элемент из дубликатов.")
    # проверим, что в основной колонке нет пустых строк. Опять же, это скорее всего будет указывать на ошибку при составлении таблицы
    for line in gold_inputs:
        assert type(line)==str and line.strip()!=""
    for line in pred_inputs:
        if (type(line)!=str and np.isnan(line)) or line.strip()=="":
            warnings.warn(f"В файле с предсказаниями, в колонке {main_column} присутствуют пустые строки, не должно быть")
    
    # считаем, сколько в основной колонке совпадающих значений. 
    # Если таблицы целиком собираются по автоматической разметке, некоторые значения могут отсутствовать в том числе и для колонки. которую мы считаем "входными данными"
    # надо посмотреть, сколько таких отсутствующих или лишних значений, и оценки считать только для того, что совпадает.
    gold_inputs = set(gold_inputs)
    pred_inputs = set(pred_inputs)
    inputs_intersection = set(gold_inputs) & set(pred_inputs)
    correct_inputs_p = np.round(100*len(inputs_intersection)/len(gold_inputs), 2)
    false_neg = gold_inputs - inputs_intersection
    false_pos = pred_inputs - inputs_intersection

    report_json[main_column+"_check"] = {}
    report_json[main_column+"_check"]["gold_lines"] = len(gold_inputs)
    report_json[main_column+"_check"]["pred_lines"] = len(pred_inputs)
    report_json[main_column+"_check"]["duplicates_in_pred"] = len(pred_inputs)!=len(set(pred_inputs))
    report_json[main_column+"_check"]["intersection"] = len(inputs_intersection)
    report_json[main_column+"_check"]["false_neg"] = len(false_neg)
    report_json[main_column+"_check"]["false_pos"] = len(false_pos)
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(report_json, f)

    print("="*50)
    print(f"Количество корректных элементов колонки {main_column}: {len(inputs_intersection)} ({correct_inputs_p}% от исходного множества)")
    print("По строкам с этими элементами будут рассчитываться оценки.")
    print(f"Количество непредсказанных элементов колонки {main_column}: {len(false_neg)}")
    print(f"Количество лишних элементов колонки {main_column}: {len(false_pos)}")

    if len(inputs_intersection)==0:
        warnings.warn(f"Количество общих элементов колонки {main_column} равно 0, дальнейший расчёт оценок не имеет смысла")
        exit()

    # теперь уже считаем оценки для указанных целевых колонок
    gold_subset = gold.loc[gold[main_column].isin(inputs_intersection)].copy()
    predict_subset = predict.loc[predict[main_column].isin(inputs_intersection)].copy()
    gold_subset.sort_values(main_column, axis=0, inplace=True)
    predict_subset.sort_values(main_column, axis=0, inplace=True)
    assert (gold_subset[main_column].values == predict_subset[main_column].values).all()
    for target_col in target_columns:
        print("="*50)
        print(f"Расчет оценок для колонки {target_col}")
        if target_col not in columns_metric_types:
            raise ValueError(f"Для колонки {target_col} не задан метод расчета оценки в словаре columns_metric_types. Либо неправильное название колонки, либо она не добавлена в словарь")
        if columns_metric_types[target_col]=="strings_set_comparison":
            recall_list = []
            precision_list = []
            f1_list = []
            # в принципе, учитывая, что выше произошла сортировка, можно и просто пройтись по zip(gold_value[target_col].values, pred_value[target_col].values), 
            # но я сначала сделал так, пусть останется
            for input_value in gold_subset[main_column].values:
                gold_value = gold_subset.loc[gold[main_column]==input_value, target_col].values[0]
                pred_value = predict_subset.loc[predict[main_column]==input_value, target_col].values[0]
                
                # если какое-то из значений nan (пустая строка), это интерпретируется тоже как элемент списка. 
                # То есть если на nan предсказано nan, то метрики 1, если что-то другое, то 0, если на что-то другое предсказано nan, то 0.
                # можно не учитывать такие строки и считать отдельно, но вряд ли имеет смысл.
                gold_value = set(str(gold_value).strip().split("; "))
                pred_value = set(str(pred_value).strip().split("; "))

                precision = len(pred_value & gold_value) /  len(pred_value)
                recall = len(pred_value & gold_value) /  len(gold_value)
                if precision + recall > 0:
                    f1 = 2*precision*recall/(precision+recall)
                else:
                    f1 = 0
                recall_list.append(recall)
                precision_list.append(precision)
                f1_list.append(f1)
            report_json[target_col] = {}
            report_json[target_col]["recall"] = np.round(100*sum(recall_list)/len(recall_list), 2)
            report_json[target_col]["precision"] = np.round(100*sum(precision_list)/len(precision_list), 2)
            report_json[target_col]["f1"] = np.round(100*sum(f1_list)/len(f1_list), 2)
            print("recall:", report_json[target_col]["recall"])
            print("precision:", report_json[target_col]["precision"])
            print("f1:", report_json[target_col]["f1"])
        elif columns_metric_types[target_col]=="classification_f1_score":
            print(classification_report(gold_subset[target_col].values, predict_subset[target_col].values))
            d = classification_report(gold_subset[target_col].values, predict_subset[target_col].values, output_dict=True)
            report_json[target_col] = d
        else:
            raise NotImplementedError(f"Не реализован метод расчета метрики {columns_metric_types[target_col]}")

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(report_json, f)