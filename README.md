# MedRecommendation
Проект по разработке рекоммендательной системы для лекарственных препаратов на основе экстрактивных подходов и генеративных.
С одной стороны интересно посмотреть, может ли генеративная сетка или другой метод 

Подготовка:
- [ ] Организовать репозиторий по шаблону cookiecutter https://drivendata.github.io/cookiecutter-data-science/
- [ ] Завести гуглдок для лит обзора, приложить сюда ссылку с доступом для чтения или комментирования.
- [ ] завести гуглтаблицу для сбора результатов экспериментов
- [ ] Прочитать статью про корпус https://arxiv.org/abs/2105.00059

Лит. обзор. Включить в него:
- [ ] Информацию про наш корпус, модели экстракции на основе Spert и нормализации.
- [ ] прочитать про генеративные модели вроде GPT-2, mT5, и другие примерно такого же размера. И как их дообучают на конкретные задачи. Выбрать одну или 2.
- [ ] Рекомендательные системы  на основе нейронок и прочего nlp.

## Экстрактивный подход.

- [x] Взять размеченный корпус, посчитать для тестовой:
  - препарат ; рейтинг ; список возможных АДР
  - симптом ; список лучших препаратов;
- [x] Взять модели и код из репозитория https://github.com/sag111/med-demo-service_react разобраться с тем, как делать предикт. Потом взять модели, обученные на тренировочной части первых фолдов. 
- [x] Сделать предикт для тестовой части корпуса и составить таблицу по предиктам.
- [x] Придумать, как эти таблицы сравнить.

Январь

- [ ] Запустить Spert на обучение на 5 фолдах и оценить результаты. Должно получиться f1 micro ~75 macro ~66; rels w/o ner micro ~65 macro ~54; rels w/ ner micro ~64 macro ~53.
- [ ] Запустить нормализацию на 5 фолдах.

Февраль

- [ ] Изменить converters/convert_input.py: добавить аргумент скрипта --use_normalization. Когда он активен, в выходном (сохраняемом) json файле в поле entities помимо полей type, start, end, text, origin_entity_id, id, было ещё поле normalization. И там было записано содержимое полей MedDRA из сущностей типа ADR и Indication исходного корпуса. Это меняется где-то между 100-180 строками. 
- [ ] Изменить converters/convert_pred.py: Добавить аргумент скрипта --use_normalization. Когда он активен, при восстановлении из формата spert в исходный формат корпуса, также используется поле normalization и предсказанные значения записываются в поле также, как и в исходном файле. Вроде это где-то между 92 и 105 строками.
- [ ] Изменить spert_eval.py так, чтобы после оценок классификации сущностей выводились оценки по нормализации.
- [ ] Провести проверку - убедиться, результаты сравнения старых файлов (для экспериментов без нормализации) не меняют значений. Потом взять файл полученный после convert_input, изменить в нём некоторые значения нормализации руками, подать в convert_pred->spert_eval, убедиться, что все считается как надо.
- [ ] Изменить spert_norm/input_reader.py, spert_norm/entities.py и spert_norm/sampling.py. Сейчас в SpertTrainer.train() после того, как вызывается JsonInputReader.read() получается экземпляр класса Dataset из entities.py, где каждый документ описывается словарем с ключами 'encodings', 'context_masks', 'entity_masks', 'entity_sizes', 'entity_types', 'rels', 'rel_masks', 'rel_types', 'entity_sample_masks', 'rel_sample_masks', надо добавить туда ключ entity_normalization, где будут содержаться индексы терминов из медры для каждой сущности. Скорее всего работа с ним будет организована в этих файлах также как с entity_types, с тем отличием, что для entity_type собирается список возможных типов, а для entity_normalization он должен быть сформирован на основе конкретного файла с терминами (файл с pt MedDRA), к которым в начало должен быть добавлин пустой термин (CONCEPT_LESS). Всем сущностям у которых было поле MedDRA должны быть поставлены индексы соответствующих терминов, остальным пустой термин (пустой термин имеет индекс 0, в корпусе adr и indication без нормализации имеют MedDRA=="", им тоже 0). Рекомендую отлаживать этот процесс в ноутбуке например как в этом: /s/ls4/users/grartem/SAG_MED/RelationExtraction/notebooks/SpertDataset_debug.ipynb или в pycharm создать небольшой скрипт для отладки  именно этих модулей. 
- [ ] Провести проверку -  превратить json в spert формате в батч с помощью классов Dataset и DataLoader, попробовать превратить обратно, убедиться, что результаты выглядят адекватно. Сохранить это как predict, прогнать через convert_pred, и spert_eval.py. получить точности близкие к 1.
- [ ] Изменить spert_trainer.py, spert_norm/models.py и spert_norm/loss.py. Должен быть добавлен дополнительный выход в модели, дополнительный лосс, и матрица весов словаря
- [ ] Запустить эксперимент на 5 фолдах на том же корпуса, но теперь с нормализацией.

Март

- [ ] Отладка, проверка, что всё работает так, как задумано. 
- [ ] анализ ошибок
- [ ] сравнительные эксперименты - сравнение оценок отдельного сперта и нормализации, последовательного (нормализация на результатах работы сперта вместо эталонных упоминаний) и совместного.
- [ ] балансировка лоссов, возможно они должны быть использованы с разными весами
- [ ] вариации использования эмбеддингов словаря и упоминаний, должны ли эмбеддинги словаря дообучаться, сейчас в обычной нормализации для эмбеддинга упоминаний используется mean pooling, в сперте max pooling. Попробовать оба варианта.
- [ ] подбор гиперпараметров. Возможно из-за доп. задачи требуется другой lr или batch size. Может другая инициализация. 
- [ ] поиск решения проблемы несбалансированности примеров для нормализации. В скриптах отдельной модели для нормализации используется только примеры у которых есть нормализация. При обучении сперта очень много примеров у которых её нет. Сначала просто сделаем, чтоб лосс игнорировал такие примеры. Но как это скажется на этапе инференса - вопрос. Может стоит использовать их в обучении.

Апрель
- [ ] Будет ясно, какие доп. эксперименты надо провести исходя из того, какие результаты получаются.
- [ ] Расширение лит. обзора
- [ ] описание устройства модели
- [ ] описание экспериментов


Май 
- [ ] написание диплома


## Генеративный подход.

Цель - получить модель, спосробную извлекать различные знания из большого неразмеченного корпуса.

Глобальные задачи:
- Сформировать юз кейсы для постановки экспериментов и процедуру оценки. сейчас есть 2 эксперимента, хорошо бы больше. Возможно подключить другие корпуса.
- Попробовать разные модели.
- попробовать разные подходы к предобучению.

Октябрь
- [x] Оформить в виде скриптов и закомитить код, использованный для прошлого эксперимента  (который с 6%). Прислать мне таблицу с колонками input;prediction;correc_answer для того эксперимента.
- [x] Проанализировать большой корпус для предобучения.
- [x] Взять большой, автоматически размеченный корпус /s/ls4/groups/g0126/fake_reviews/ner_800k_jsonl.zip. Проверить, как много там про косметику (которая нас мало интересует), а не про лекарства. Смотреть на размеченные там drugname и drugclass.
- [x] Составить гистограмму - на сколько отзывов сколько препаратов приходится. Это надо, чтобы определиться, нужно ли добирать корпус.
- [ ] Попробовать подобрать промты к gigachad/chatgpt для генерации рекомендаций. Оценить их по нашим тестам, сохранить их выходы.

Ноябрь
- [ ] Попробовать дообучить модель t5 на размеченном корпусе (3800 отзывов) на unsupervised задачу (предсказание следующего токена / Или предсказание заголовка или общего впечатления по тексту отзыва). 
- [ ] Протестировать её также как и не предобученную.
Попробовать дообучить модель на semi-supervised (учесть разметку), например предсказывать список адров или симптомов по предложению с препаратов, или BNE-Pos/Worse/ADE_Neg по препарату и симптомам. То есть учитывать только полезные предложения. 

Декабрь 
- [ ] Эксперименты с gpt2 моделью такие же как с t5.

Январь 
- [ ] Сформировать 4 таблицы с выборками и разделение на фолды, и проверить: сиптом-препараты, симптом-хорошие препараты, препарат-адры, препарат-класс.
- [ ] Обучить ruT5 (опционально FRED), GPT-2 на предасказание по этим таблица (инпут одна колонка, аутпут - вторая)
- [ ] Предобучить ruT5 unsupervised/semi-supervised на корпусе 3800. Задача - предсказание замаскированных спанов. Спаны маскировать либо те, которые выделены в разметке (тогда semi-supervised) либо случайные.
- [ ] На выходе должны быть точности по 5 фолдам для каждой таблицы для моделей: ruT5, GPT-2, ruT5-pretrained

Февраль 
- [ ] Предобучение ruT5 на большом корпусе - также как и на 3800, но на 200к (отфильтрованные из 700к)
- [ ] Предобучение GPT-2 (на малом, потом на большом)
- [ ] исправление ошибок, думаем, как еще улучшить точность

Март

Апрель

Май - июнь
- [ ] Написание диплома, перепроверка экспериментов


Дальнейшие планы:
- расширение корпуса
- расширить use case 
