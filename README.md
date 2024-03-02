<h1>
HAY 2 TIPOS DE JUEGOS CONVERSACIONALES:

Juego de texto

Juego de conversación

</h1>
# juego-conversacional modo texto
## Ejecución
python chat.py [prompt system] [saludo inicial del asistente]

## Ejemplo de juego conversacional PROACTIVO
### Ejemplo 1:
```bash
python chat.py --short "You are the CEO of a consulting company called mAgIc, dedicated to finding artificial intelligence solutions for companies. You are interviewing a candidate to work for the company, you are interested in programming skills, mathematics, data science, AI and especially NLP. Ask questions about all this and probe. If the candidate says something wrong let them know." "Good morning, have a seat, tell me your name."

# En español con traducción al inglés bidireccional (parametro "-es")
python chat.py -es --short "You are the CEO of a consulting company called mAgIc, dedicated to finding artificial intelligence solutions for companies. You are interviewing a candidate to work for the company, you are interested in programming skills, mathematics, data science, AI and especially NLP. Ask questions about all this and probe. If the candidate says something wrong let them know. After 4 questions if most were good responses you must say 'CONGRATULATION YOU ARE IN OUR TEAM!!' if there was a bad response then you must say 'The interview is finished, thanks for come' IMPORTANT: if the interviewee is rude then you must say 'go away, close the door when go out'" "Good morning, have a seat, tell me your name."

 # En español (funciona mejor en inglés o con traducción al inglés bidireccional)
python chat.py --short "Eres el CEO de una empresa consultora llamada mAgIc, dedicada a buscar soluciones de inteligencia artificial para las empresas. Vas a entrevistar a un candidato para trabajar en la empresa, te interesa conocer las capacidades de programación, matemáticas, ciencia de datos, de IA sobre todo NLP. Haz preguntas sobre todo esto e indaga. Si el candidato dice algo erroneo hazselo saber. Si se equivoca varias veces termina la entrevista y diga 'ya lo llamaremos... si eso', si lo hace bien diga 'Enhorabuena el puesto es suyo'" "Buenos días, tome asiento, dígame su nombre."
```

### ejemplo2:
```bash
python chat.py --short "I want us to play a roleplay where you are a police officer who interrogates me about why I was at a crime scene. The scene is a pub where there were a few people at the bar: 1 couple and a group of 4 friends and me who was alone. A murderer entered and killed the couple with two shots, then ran away and got on his motorcycle that was parked at the door and fled. The police arrived 10 minutes later and that is where the interrogation begins. You are a police inspector and you want information about what happened and also to verify that I am not a suspect." "Good morning, have a seat, tell me your name."

 #En español (funciona mejor en inglés)
python chat.py --short "Quiero que juguemos a un juego de rol en el que tú eres un agente de policía que me interroga sobre por qué estaba en la escena de un crimen. La escena es un pub donde había unas cuantas personas en la barra: 1 pareja y un grupo de 4 amigos y yo que estaba solo. Un asesino entró y mató a la pareja de dos tiros, luego salió corriendo y se subió a su moto que estaba aparcada en la puerta y huyó. La policía llegó 10 minutos después y ahí empieza el interrogatorio. Usted es inspector de policía y quiere información sobre lo ocurrido y también verificar que no soy sospechoso." "Buenos días, tome asiento, dígame su nombre".
```

## Ejemplo de juego conversacional RECEPTIVO
```bash
python chat.py --short "I want us to play a roleplay where I am a police officer who interrogate you about why you were at a crime scene. The scene is a pub where there were a few people at the bar: 1 couple and a group of 4 friends and you who were alone. A murderer entered and killed the couple with two shots, then ran away and got on his motorcycle that was parked at the door and fled. The police arrived 10 minutes later and that is where the interrogation begins. I am a police inspector and I want information about what happened and also to verify that you are not a suspect.You are Peter, a medical student who had gone down to the bar to rest after a long day of studying." "Hi, Mr Inspector, I am a bit nervous."

# En español (traducción bidirecciona)
python chat.py --short -es "I want us to play a roleplay where I am a police officer who interrogate you about why you were at a crime scene. The scene is a pub where there were a few people at the bar: 1 couple and a group of 4 friends and you who were alone. A murderer entered and killed the couple with two shots, then ran away and got on his motorcycle that was parked at the door and fled. The police arrived 10 minutes later and that is where the interrogation begins. I am a police inspector and I want information about what happened and also to verify that you are not a suspect.You are Peter, a medical student who had gone down to the bar to rest after a long day of studying." "Hi, Mr Inspector, I am a bit nervous."

 #En español de base (funciona mejor en inglés)
python chat.py --short "Quiero que juguemos a un juego de rol en el que yo soy un agente de policía que te interroga sobre por qué estabas en la escena de un crimen. La escena es un pub donde había unas cuantas personas en el bar: 1 pareja y un grupo de 4 amigos y tú que estabas solo. Un asesino entra y mata a la pareja de dos tiros, luego sale corriendo y se sube a su moto que estaba aparcada en la puerta y huye. La policía llegó 10 minutos después y ahí empieza el interrogatorio. Soy inspector de policía y quiero información sobre lo sucedido y también verificar que usted no es sospechoso.Usted es Peter, un estudiante de medicina que había bajado al bar a descansar después de un largo día de estudio." "Hola, señor inspector, estoy un poco nervioso".
```

# Juego conversacional

## En inglés.
```bash
python server_zephyr.py --short "You are the CEO of a consulting company called mAgIc, dedicated to finding artificial intelligence solutions for companies. You are interviewing a candidate to work for the company, you are interested in programming skills, mathematics, data science, AI and especially NLP. Ask questions about all this and probe. If the candidate says something wrong let them know. After 4 questions if most were good responses you must say 'CONGRATULATION YOU ARE IN OUR TEAM!!' if there was a bad response then you must say 'The interview is finished, thanks for come.' IMPORTANT: if the interviewee is rude or cocky or is insulting then you must say 'go away! close on your way out!'" "Good morning, have a seat, tell me your name."
```

## En español (te va traduciendo todo lo que dices al inglés y lo que dice el chatbot al español)
```bash
python server_zephyr.py -es --short "You are the CEO of a consulting company called mAgIc, dedicated to finding artificial intelligence solutions for companies. You are interviewing a candidate to work for the company, you are interested in programming skills, mathematics, data science, AI and especially NLP. Ask questions about all this and probe. If the candidate says something wrong let them know. After 4 questions if most were good responses you must say 'CONGRATULATION YOU ARE IN OUR TEAM!!' if there was a bad response then you must say 'The interview is finished, thanks for come' IMPORTANT: if the interviewee is rude then you must say 'go away, close the door when go out'" "Good morning, have a seat, tell me your name."
```



## citaciones

Este proyecto utiliza `fairseq`, un toolkit de síntesis de voz escalable e integrable:
@inproceedings{wang-etal-2021-fairseq,
    title = "fairseq S{\^{}}2: A Scalable and Integrable Speech Synthesis Toolkit",
    author = "Wang, Changhan  and
      Hsu, Wei-Ning  and
      Adi, Yossi  and
      Polyak, Adam  and
      Lee, Ann  and
      Chen, Peng-Jen  and
      Gu, Jiatao  and
      Pino, Juan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.17",
    doi = "10.18653/v1/2021.emnlp-demo.17",
    pages = "143--152",
}


También un transcriptor de voz a texto llamado Whisper:
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}


Y Un LLM de generación de texto basado en mistral 7b llamado Zephyr:

@misc{tunstall2023zephyr,
      title={Zephyr: Direct Distillation of LM Alignment}, 
      author={Lewis Tunstall and Edward Beeching and Nathan Lambert and Nazneen Rajani and Kashif Rasul and Younes Belkada and Shengyi Huang and Leandro von Werra and Clémentine Fourrier and Nathan Habib and Nathan Sarrazin and Omar Sanseviero and Alexander M. Rush and Thomas Wolf},
      year={2023},
      eprint={2310.16944},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


## citación

@software{javier_gimenez_moya_2024_tu_proyecto,
    title = "{Juegos Conversacionales}: Herramienta educativa para aprender a hablar inglés mediante Juegos conversacionales guiados por un chatbot de voz y texto",
    author = "Giménez Moya, Javier",
    month = mar,
    year = "2024",
    version = "1.0",
    publisher = "{Trabajo Fin de Master en VIU (Universidad Internacional de Valencia)}",
    address = "Águilas, Spain",
    howpublished = "\url{https://github.com/javileyes/juego-conversacional}"
}