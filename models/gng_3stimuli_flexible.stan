data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=1, upper=3> cue[N, T];
  int<lower=-1, upper=1> pressed[N, T];
  real outcome[N, T];
}

transformed data {
  vector[3] initV;
  initV = rep_vector(0.0, 3);
}

parameters {
  // declare as vectors for vectorizing
  vector[7] mu_pr;
  vector<lower=0>[7] sigma;
  vector[N] xi_pr;        // noise
  vector[N] ep_pr;        // learning rate
  vector[N] b_pr;         // go bias
  vector[N] rho_pr;       // rho, inv temp
  vector[N] beta_pr;        // learning rate multiplier for associability
  vector[N] kappa_pr;        // scaling factor for w  
  vector[N] assoc0_pr;        // initial associability
}

transformed parameters {
  vector<lower=0, upper=1>[N] xi;
  vector<lower=0, upper=1>[N] ep;
  vector[N] b;
  vector<lower=0>[N] rho;
  vector<lower=0, upper=1>[N] beta;
  vector<lower=0>[N] kappa;
  vector<lower=0>[N] assoc0;  

  for (i in 1:N) {
    xi[i] = Phi_approx(mu_pr[1] + sigma[1] * xi_pr[i]);
    ep[i] = Phi_approx(mu_pr[2] + sigma[2] * ep_pr[i]);
    beta[i] = Phi_approx(mu_pr[5] + sigma[5] * beta_pr[i]);
  }
  b   = mu_pr[3] + sigma[3] * b_pr; // vectorization
  rho = exp(mu_pr[4] + sigma[4] * rho_pr);
  kappa = exp(mu_pr[6] + sigma[6] * kappa_pr);
  assoc0 = mu_pr[7] + sigma[7] * assoc0_pr;
}

model {
// gng_m4: RW(rew/pun) + noise + bias + pi model (M5 in Cavanagh et al 2013 J Neuro)
  // hyper parameters
  mu_pr[1]  ~ normal(0, 1.0);
  mu_pr[2]  ~ normal(0, 1.0);
  mu_pr[3]  ~ normal(0, 10.0);
  mu_pr[4]  ~ normal(0, 1.0);
  mu_pr[5]  ~ normal(0, 1.0);
  mu_pr[6]  ~ normal(0, 1.0);
  mu_pr[7]  ~ normal(0, 1.0);
  sigma[1] ~ normal(0, 0.2);
  sigma[2] ~ normal(0, 0.2);
  sigma[3] ~ cauchy(0, 1.0);
  sigma[4]   ~ normal(0, 0.2);
  sigma[5]   ~ normal(0, 0.2);
  sigma[6]   ~ normal(0, 0.2);
  sigma[7] ~ cauchy(0, 1.0);

  // individual parameters w/ Matt trick
  xi_pr  ~ normal(0, 1.0);
  ep_pr  ~ normal(0, 1.0);
  b_pr   ~ normal(0, 1.0);
  rho_pr ~ normal(0, 1.0);
  beta_pr ~ normal(0, 1.0);
  kappa_pr ~ normal(0, 1.0);
  assoc0_pr  ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[3] wv_g;  // action weight for go
    vector[3] wv_ng; // action weight for nogo
    vector[3] qv_g;  // Q value for go
    vector[3] qv_ng; // Q value for nogo
    vector[3] sv;    // stimulus value
    vector[3] pGo;   // prob of go (press)

    real pi; // PIT parameter flexible
    real assoc; // assoc value which is updated each trial
    real absRPE; // absRPE each trial

  
    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;
    assoc = assoc0[i];
    
    for (t in 1:Tsubj[i]) {
      pi = 1/(1+exp(-kappa[i]*(assoc - assoc0[i])));
      // if (kappa[i]*assoc < 1.0){
      //   pi = kappa[i]*assoc;
      // }
      // else {
      //   pi = 1.0;
      // }
      wv_g[cue[i, t]]  = (1-pi) * qv_g[cue[i, t]] + b[i] + pi * sv[cue[i, t]];
      wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  // qv_ng is always equal to wv_ng (regardless of action)
      pGo[cue[i, t]]   = inv_logit(wv_g[cue[i, t]] - wv_ng[cue[i, t]]);
      {  // noise
        pGo[cue[i, t]]   *= (1 - xi[i]);
        pGo[cue[i, t]]   += xi[i]/2;
      }
      pressed[i, t] ~ bernoulli(pGo[cue[i, t]]);

      // after receiving feedback, update sv[t + 1]
      sv[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - sv[cue[i, t]]);
      
      // update action values
      if (pressed[i, t]) { // update go value
        qv_g[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - qv_g[cue[i, t]]);
        absRPE = fabs(rho[i] * outcome[i, t] - qv_g[cue[i, t]]);
      } else { // update no-go value
        qv_ng[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - qv_ng[cue[i, t]]);
        absRPE = fabs(rho[i] * outcome[i, t] - qv_ng[cue[i, t]]);
      }
      assoc += beta[i]*ep[i]*(absRPE - assoc); 
    } // end of t loop
  } // end of i loop
}

generated quantities {
  real<lower=0, upper=1> mu_xi;
  real<lower=0, upper=1> mu_ep;
  real mu_b;
  real<lower=0> mu_rho;
  real<lower=0, upper=1> mu_beta; 
  real<lower=0> mu_kappa;
  real<lower=0> mu_assoc0;  

  real log_lik[N];
  real Qgo[N, T];
  real Qnogo[N, T];
  real Wgo[N, T];
  real Wnogo[N, T];
  real SV[N, T];
  real Assoc_Arr[N, T];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_xi  = Phi_approx(mu_pr[1]);
  mu_ep  = Phi_approx(mu_pr[2]);
  mu_b   = mu_pr[3];
  mu_rho = exp(mu_pr[4]);
  mu_beta = Phi_approx(mu_pr[5]);
  mu_kappa = exp(mu_pr[6]);
  mu_assoc0 = mu_pr[7];

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[3] wv_g;  // action weight for go
      vector[3] wv_ng; // action weight for nogo
      vector[3] qv_g;  // Q value for go
      vector[3] qv_ng; // Q value for nogo
      vector[3] sv;    // stimulus value
      vector[3] pGo;   // prob of go (press)

      real pi; // PIT parameter flexible
      real assoc; // assoc value which is updated each trial
      real absRPE; // absRPE each trial

      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;
      assoc = assoc0[i];
      log_lik[i] = 0;

      for (t in 1:Tsubj[i]) {
        pi = 1/(1+exp(-kappa[i]*(assoc - assoc0[i])));
        // if (kappa[i]*assoc < 1.0){
        //   pi = kappa[i]*assoc;
        // }
        // else {
        //   pi = 1.0;
        // }
        wv_g[cue[i, t]]  = (1-pi) * qv_g[cue[i, t]] + b[i] + pi * sv[cue[i, t]];
        wv_ng[cue[i, t]] = qv_ng[cue[i, t]];  // qv_ng is always equal to wv_ng (regardless of action)
        pGo[cue[i, t]]   = inv_logit(wv_g[cue[i, t]] - wv_ng[cue[i, t]]);
        {  // noise
          pGo[cue[i, t]]   *= (1 - xi[i]);
          pGo[cue[i, t]]   += xi[i]/2;
        }
        log_lik[i] += bernoulli_lpmf(pressed[i, t] | pGo[cue[i, t]]);

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(pGo[cue[i, t]]);

        // Model regressors --> store values before being updated
        Qgo[i, t]   = qv_g[cue[i, t]];
        Qnogo[i, t] = qv_ng[cue[i, t]];
        Wgo[i, t]   = wv_g[cue[i, t]];
        Wnogo[i, t] = wv_ng[cue[i, t]];
        SV[i, t]    = sv[cue[i, t]];
        Assoc_Arr[i, t] = assoc;
        // after receiving feedback, update sv[t + 1]
        sv[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - sv[cue[i, t]]);

        // update action values
        if (pressed[i, t]) { // update go value
          qv_g[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - qv_g[cue[i, t]]);
          absRPE = fabs(rho[i] * outcome[i, t] - qv_g[cue[i, t]]);
        } else { // update no-go value
          qv_ng[cue[i, t]] += ep[i] * (rho[i] * outcome[i, t] - qv_ng[cue[i, t]]);
          absRPE = fabs(rho[i] * outcome[i, t] - qv_ng[cue[i, t]]);
        }
        assoc += beta[i]*ep[i]*(absRPE - assoc); 

      } // end of t loop
    } // end of i loop
  } // end of local section
}

