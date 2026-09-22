# ============================================================
# BAYESIAN LOGISTIC REGRESSION
# Predicting ADHD status from primary CFG measures
# ============================================================

# 1. Packages ----

library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(bayestestR)

set.seed(2026)

options(mc.cores = parallel::detectCores())

# 2. Load data ----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)

# 3. Prepare outcome and predictors ----

df_logistic <- df %>%
  transmute(
    
    # Outcome:
    # Without ADHD = 0
    # ADHD = 1
    ADHD_status = case_when(
      diva_group == "TD"   ~ 0,
      diva_group == "ADHD" ~ 1,
      TRUE                 ~ NA_real_
    ),
    
    # Primary CFG measures
    originality = `Gallery Orig`,
    fluency = `#galleries`,
    g = g_empirical,
    alpha = alpha_empirical,
    efficiency_exploration = `exp efficiency`,
    efficiency_exploitation = `scav efficiency`
  )

# 3.1 Missing values ----

df_logistic %>%
  summarise(
    across(
      everything(),
      ~ sum(is.na(.x))
    )
  )

df_logistic <- df_logistic %>%
  drop_na()

# 4. Standardize predictors ----

df_logistic <- df_logistic %>%
  mutate(
    across(
      c(
        originality,
        fluency,
        g,
        alpha,
        efficiency_exploration,
        efficiency_exploitation
      ),
      ~ as.numeric(scale(.x)),
      .names = "{.col}_z"
    )
  )

# 5. Correlations among predictors ----

df_logistic %>%
  select(
    originality,
    fluency,
    g,
    alpha,
    efficiency_exploration,
    efficiency_exploitation
  ) %>%
  cor(
    use = "pairwise.complete.obs",
    method = "spearman"
  ) %>%
  
  
  # 6. Priors ----

priors_logistic <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1.5), class = "Intercept")
)

# 7. Bayesian logistic regression ----

m_adhd_logistic <- brm(
  ADHD_status ~
    originality_z +

    fluency_z +
    g_z +
    alpha_z +
    efficiency_exploration_z +
    efficiency_exploitation_z,

  data = df_logistic,
  
  family = bernoulli(link = "logit"),
  
  prior = priors_logistic,
  
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  
  backend = "cmdstanr",
  seed = 2026,
  
  control = list(
    adapt_delta = 0.95
  )
)

summary(m_adhd_logistic)

# 8. Posterior summary ----

adhd_logistic_posterior <- describe_posterior(
  m_adhd_logistic,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

adhd_logistic_posterior

