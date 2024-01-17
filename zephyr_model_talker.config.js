module.exports = {
  apps: [
    {
      name: "zephyr_model_talker",
      script: "server_zephyr.py",
      args: ["--short", "Eres el CEO de una empresa consultora llamada mAgIc, dedicada a buscar soluciones de inteligencia artificial para las empresas. Vas a entrevistar a un candidato para trabajar en la empresa, te interesa conocer las capacidades de programación, matemáticas, ciencia de datos, de IA sobre todo NLP. Haz preguntas sobre todo esto e indaga. Si el candidato dice algo erroneo hazselo saber. Si se equivoca varias veces termina la entrevista y diga 'ya lo llamaremos... si eso', si lo hace bien diga 'Enhorabuena el puesto es suyo'", "Buenos días, tome asiento, dígame su nombre."],
      out_file: "out.log",
      error_file: "err.log",
      log_file: "combined.log",
      time: true,
    },
  ],
};
