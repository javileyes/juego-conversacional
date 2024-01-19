module.exports = {
  apps: [
    {
      name: "phi_model_talker",
      script: "server_phi_1_5.py",
      out_file: "out.log",
      error_file: "err.log",
      log_file: "combined.log",
      time: true,
    },
  ],
};
