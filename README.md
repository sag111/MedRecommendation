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

Экстрактивный подход.
- [ ] Взять размеченный корпус, посчитать для тестовой:
  - препарат ; рейтинг ; список возможных АДР
  - симптом ; список лучших препаратов;
- [ ] Взять модели и код из репозитория https://github.com/sag111/med-demo-service_react разобраться с тем, как делать предикт. Потом взять модели, обученные на тренировочной части первых фолдов. 
- [ ] Сделать предикт для тестовой части корпуса и составить таблицу по предиктам.
- [ ] Придумать, как эти таблицы сравнить.

## Генеративный подход.

Цель - получить модель, спосробную извлекать различные знания из большого неразмеченного корпуса.

Глобальные задачи:
- Сформировать юз кейсы для постановки экспериментов и процедуру оценки. сейчас есть 2 эксперимента, хорошо бы больше. Возможно подключить другие корпуса.
- Попробовать разные модели.
- попробовать разные подходы к предобучению.

Октябрь
- [ ] Оформить в виде скриптов и закомитить код, использованный для прошлого эксперимента  (который с 6%). Прислать мне таблицу с колонками input;prediction;correc_answer для того эксперимента.
- [ ] Проанализировать большой корпус для предобучения.
- [ ] Взять большой, автоматически размеченный корпус /s/ls4/groups/g0126/fake_reviews/ner_800k_jsonl.zip. Проверить, как много там про косметику (которая нас мало интересует), а не про лекарства. Смотреть на размеченные там drugname и drugclass.
- [ ] Составить гистограмму - на сколько отзывов сколько препаратов приходится. Это надо, чтобы определиться, нужно ли добирать корпус.
- [ ] Попробовать подобрать промты к gigachad/chatgpt для генерации рекомендаций. Оценить их по нашим тестам, сохранить их выходы.

Ноябрь
- [ ] Попробовать дообучить модель t5 на размеченном корпусе (3800 отзывов) на unsupervised задачу (предсказание следующего токена / Или предсказание заголовка или общего впечатления по тексту отзыва). 
- [ ] Протестировать её также как и не предобученную.
Попробовать дообучить модель на semi-supervised (учесть разметку), например предсказывать список адров или симптомов по предложению с препаратов, или BNE-Pos/Worse/ADE_Neg по препарату и симптомам. То есть учитывать только полезные предложения. 

Декабрь 
- [ ] Эксперименты с gpt2 моделью такие же как с t5.

Январь 
- [ ] Предобучение на большом корпусе. supervised и semisupervised.

Февраль 

Март

Апрель

Май - июнь
- [ ] Написание диплома, перепроверка экспериментов


Дальнейшие планы:
- расширение корпуса
- расширить use case 
