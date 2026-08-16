
# Load the final dataset
df_creativity_ADHD <- readr::read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)

# Create a folder for fitted Bayesian models
dir.create(
  "Analysis/models",
  recursive = TRUE,
  showWarnings = FALSE
)

# Confirm that the folder exists
print(
  dir.exists("Analysis/models")
)


# ---- Gender distribution by group ----

# Counts by group, including "other" and missing values
gender_counts <- table(
  Group = factor(
    df_creativity_ADHD$diva_group,
    levels = c("TD", "ADHD")
  ),
  Gender = df_creativity_ADHD$gender,
  useNA = "ifany"
)

print(gender_counts)

# ---- Age by group ----

# Keep participants with non-missing age
age_data <- subset(
  df_creativity_ADHD,
  !is.na(age) &
    diva_group %in% c("TD", "ADHD")
)

# Descriptive statistics
age_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_age <- age_data$age[
      age_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_age),
      Mean = mean(group_age),
      SD = sd(group_age)
    )
  })
)

age_descriptives$Mean <- round(age_descriptives$Mean, 2)
age_descriptives$SD <- round(age_descriptives$SD, 2)

print(age_descriptives)

# ---- Bayesian analysis of age ----

# Set group order
age_data$diva_group <- factor(
  age_data$diva_group,
  levels = c("TD", "ADHD")
)

# Standardize age for the Bayesian model
age_center <- mean(age_data$age)
age_scale <- sd(age_data$age)

age_data$age_z <- (
  age_data$age - age_center
) / age_scale

# Bayesian model:
# separate means and standard deviations for TD and ADHD
age_model <- brm(
  formula = bf(
    age_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = age_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(4, parallel::detectCores()),
  seed = 20260813,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/age_model",
  file_refit = "on_change"
)

# Extract posterior draws
age_draws <- posterior::as_draws_df(age_model)

# Posterior group means, transformed back to years
age_mean_TD <- (
  age_draws$b_diva_groupTD * age_scale
) + age_center

age_mean_ADHD <- (
  age_draws$b_diva_groupADHD * age_scale
) + age_center

# Mean difference: ADHD minus TD
age_mean_difference <- (
  age_mean_ADHD - age_mean_TD
)

# Posterior standard deviations, transformed back to years
age_sd_TD <- exp(
  age_draws$b_sigma_diva_groupTD
) * age_scale

age_sd_ADHD <- exp(
  age_draws$b_sigma_diva_groupADHD
) * age_scale

# Posterior pooled standard deviation
n_TD <- sum(age_data$diva_group == "TD")
n_ADHD <- sum(age_data$diva_group == "ADHD")

age_pooled_sd <- sqrt(
  (
    (n_TD - 1) * age_sd_TD^2 +
      (n_ADHD - 1) * age_sd_ADHD^2
  ) /
    (n_TD + n_ADHD - 2)
)

# Posterior Cohen's d: ADHD minus TD
age_cohens_d <- (
  age_mean_difference / age_pooled_sd
)

# Function for posterior estimate and central 95% credible interval
posterior_result <- function(x) {
  
  c(
    Estimate = mean(x),
    CrI_95_low = unname(
      quantile(x, 0.025)
    ),
    CrI_95_high = unname(
      quantile(x, 0.975)
    )
  )
}

age_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(age_mean_difference),
  
  "Cohen's d" =
    posterior_result(age_cohens_d)
)

print(
  round(age_bayesian_results, 3)
)

# ---- ASRS by group ----

asrs_data <- subset(
  df_creativity_ADHD,
  !is.na(asrs) &
    diva_group %in% c("TD", "ADHD")
)

asrs_data$diva_group <- factor(
  asrs_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
asrs_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_asrs <- asrs_data$asrs[
      asrs_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_asrs),
      Mean = mean(group_asrs),
      SD = sd(group_asrs)
    )
  })
)

asrs_descriptives$Mean <- round(
  asrs_descriptives$Mean,
  2
)

asrs_descriptives$SD <- round(
  asrs_descriptives$SD,
  2
)

print(asrs_descriptives)

# ---- Bayesian analysis of ASRS ----

# Standardize ASRS for the Bayesian model
asrs_center <- mean(asrs_data$asrs)
asrs_scale <- sd(asrs_data$asrs)

asrs_data$asrs_z <- (
  asrs_data$asrs - asrs_center
) / asrs_scale

# Fit and save the Bayesian model
asrs_model <- brm(
  formula = bf(
    asrs_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = asrs_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260814,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/asrs_model",
  file_refit = "on_change"
)

# Check model convergence
print(asrs_model)

# Extract posterior draws
asrs_draws <- posterior::as_draws_df(
  asrs_model
)

# Posterior group means on the original ASRS scale
asrs_mean_TD <- (
  asrs_draws$b_diva_groupTD *
    asrs_scale
) + asrs_center

asrs_mean_ADHD <- (
  asrs_draws$b_diva_groupADHD *
    asrs_scale
) + asrs_center

# Posterior mean difference: ADHD minus TD
asrs_mean_difference <- (
  asrs_mean_ADHD - asrs_mean_TD
)

# Posterior group standard deviations
asrs_sd_TD <- exp(
  asrs_draws$b_sigma_diva_groupTD
) * asrs_scale

asrs_sd_ADHD <- exp(
  asrs_draws$b_sigma_diva_groupADHD
) * asrs_scale

# Group sample sizes
n_TD_asrs <- sum(
  asrs_data$diva_group == "TD"
)

n_ADHD_asrs <- sum(
  asrs_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
asrs_pooled_sd <- sqrt(
  (
    (n_TD_asrs - 1) * asrs_sd_TD^2 +
      (n_ADHD_asrs - 1) * asrs_sd_ADHD^2
  ) /
    (n_TD_asrs + n_ADHD_asrs - 2)
)

# Posterior Cohen's d: ADHD minus TD
asrs_cohens_d <- (
  asrs_mean_difference /
    asrs_pooled_sd
)

# Summarize posterior results
asrs_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      asrs_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      asrs_cohens_d
    )
)

print(
  round(asrs_bayesian_results, 3)
)

# ---- WURS by group ----

wurs_data <- subset(
  df_creativity_ADHD,
  !is.na(wurs) &
    diva_group %in% c("TD", "ADHD")
)

wurs_data$diva_group <- factor(
  wurs_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
wurs_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_wurs <- wurs_data$wurs[
      wurs_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_wurs),
      Mean = mean(group_wurs),
      SD = sd(group_wurs)
    )
  })
)

wurs_descriptives$Mean <- round(
  wurs_descriptives$Mean,
  2
)

wurs_descriptives$SD <- round(
  wurs_descriptives$SD,
  2
)

print(wurs_descriptives)

print(wurs_model)

# Extract posterior draws
wurs_draws <- posterior::as_draws_df(
  wurs_model
)

# Posterior group means on the original WURS scale
wurs_mean_TD <- (
  wurs_draws$b_diva_groupTD *
    wurs_scale
) + wurs_center

wurs_mean_ADHD <- (
  wurs_draws$b_diva_groupADHD *
    wurs_scale
) + wurs_center

# Posterior mean difference: ADHD minus TD
wurs_mean_difference <- (
  wurs_mean_ADHD - wurs_mean_TD
)

# Posterior group standard deviations
wurs_sd_TD <- exp(
  wurs_draws$b_sigma_diva_groupTD
) * wurs_scale

wurs_sd_ADHD <- exp(
  wurs_draws$b_sigma_diva_groupADHD
) * wurs_scale

# Group sample sizes
n_TD_wurs <- sum(
  wurs_data$diva_group == "TD"
)

n_ADHD_wurs <- sum(
  wurs_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
wurs_pooled_sd <- sqrt(
  (
    (n_TD_wurs - 1) * wurs_sd_TD^2 +
      (n_ADHD_wurs - 1) * wurs_sd_ADHD^2
  ) /
    (n_TD_wurs + n_ADHD_wurs - 2)
)

# Posterior Cohen's d: ADHD minus TD
wurs_cohens_d <- (
  wurs_mean_difference /
    wurs_pooled_sd
)

# Summarize posterior results
wurs_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      wurs_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      wurs_cohens_d
    )
)

print(
  round(wurs_bayesian_results, 3)
)

# ---- STAI by group ----

stai_data <- subset(
  df_creativity_ADHD,
  !is.na(stai) &
    diva_group %in% c("TD", "ADHD")
)

stai_data$diva_group <- factor(
  stai_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
stai_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_stai <- stai_data$stai[
      stai_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_stai),
      Mean = mean(group_stai),
      SD = sd(group_stai)
    )
  })
)

stai_descriptives$Mean <- round(
  stai_descriptives$Mean,
  2
)

stai_descriptives$SD <- round(
  stai_descriptives$SD,
  2
)

print(stai_descriptives)

# ---- Bayesian analysis of STAI ----

# Standardize STAI for the Bayesian model
stai_center <- mean(stai_data$stai)
stai_scale <- sd(stai_data$stai)

stai_data$stai_z <- (
  stai_data$stai - stai_center
) / stai_scale

# Fit and save the Bayesian model
stai_model <- brm(
  formula = bf(
    stai_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = stai_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260816,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/stai_model",
  file_refit = "on_change"
)

# Check model convergence
print(stai_model)

# Extract posterior draws
stai_draws <- posterior::as_draws_df(
  stai_model
)

# Posterior group means on the original STAI scale
stai_mean_TD <- (
  stai_draws$b_diva_groupTD *
    stai_scale
) + stai_center

stai_mean_ADHD <- (
  stai_draws$b_diva_groupADHD *
    stai_scale
) + stai_center

# Posterior mean difference: ADHD minus TD
stai_mean_difference <- (
  stai_mean_ADHD - stai_mean_TD
)

# Posterior group standard deviations
stai_sd_TD <- exp(
  stai_draws$b_sigma_diva_groupTD
) * stai_scale

stai_sd_ADHD <- exp(
  stai_draws$b_sigma_diva_groupADHD
) * stai_scale

# Group sample sizes
n_TD_stai <- sum(
  stai_data$diva_group == "TD"
)

n_ADHD_stai <- sum(
  stai_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
stai_pooled_sd <- sqrt(
  (
    (n_TD_stai - 1) * stai_sd_TD^2 +
      (n_ADHD_stai - 1) * stai_sd_ADHD^2
  ) /
    (n_TD_stai + n_ADHD_stai - 2)
)

# Posterior Cohen's d: ADHD minus TD
stai_cohens_d <- (
  stai_mean_difference /
    stai_pooled_sd
)

# Summarize posterior results
stai_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      stai_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      stai_cohens_d
    )
)

print(
  round(stai_bayesian_results, 3)
)

# ---- BDI by group ----

bdi_data <- subset(
  df_creativity_ADHD,
  !is.na(bdi) &
    diva_group %in% c("TD", "ADHD")
)

bdi_data$diva_group <- factor(
  bdi_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
bdi_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_bdi <- bdi_data$bdi[
      bdi_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_bdi),
      Mean = mean(group_bdi),
      SD = sd(group_bdi)
    )
  })
)

bdi_descriptives$Mean <- round(
  bdi_descriptives$Mean,
  2
)

bdi_descriptives$SD <- round(
  bdi_descriptives$SD,
  2
)

print(bdi_descriptives)

# ---- Bayesian analysis of BDI ----

# Standardize BDI for the Bayesian model
bdi_center <- mean(bdi_data$bdi)
bdi_scale <- sd(bdi_data$bdi)

bdi_data$bdi_z <- (
  bdi_data$bdi - bdi_center
) / bdi_scale

# Fit and save the Bayesian model
bdi_model <- brm(
  formula = bf(
    bdi_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = bdi_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260817,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/bdi_model",
  file_refit = "on_change"
)

# Check model convergence
print(bdi_model)

# Extract posterior draws
bdi_draws <- posterior::as_draws_df(
  bdi_model
)

# Posterior group means on the original BDI scale
bdi_mean_TD <- (
  bdi_draws$b_diva_groupTD *
    bdi_scale
) + bdi_center

bdi_mean_ADHD <- (
  bdi_draws$b_diva_groupADHD *
    bdi_scale
) + bdi_center

# Posterior mean difference: ADHD minus TD
bdi_mean_difference <- (
  bdi_mean_ADHD - bdi_mean_TD
)

# Posterior group standard deviations
bdi_sd_TD <- exp(
  bdi_draws$b_sigma_diva_groupTD
) * bdi_scale

bdi_sd_ADHD <- exp(
  bdi_draws$b_sigma_diva_groupADHD
) * bdi_scale

# Group sample sizes
n_TD_bdi <- sum(
  bdi_data$diva_group == "TD"
)

n_ADHD_bdi <- sum(
  bdi_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
bdi_pooled_sd <- sqrt(
  (
    (n_TD_bdi - 1) * bdi_sd_TD^2 +
      (n_ADHD_bdi - 1) * bdi_sd_ADHD^2
  ) /
    (n_TD_bdi + n_ADHD_bdi - 2)
)

# Posterior Cohen's d: ADHD minus TD
bdi_cohens_d <- (
  bdi_mean_difference /
    bdi_pooled_sd
)

# Summarize posterior results
bdi_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      bdi_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      bdi_cohens_d
    )
)

print(
  round(bdi_bayesian_results, 3)
)

# ---- OCI-R by group ----

ocir_data <- subset(
  df_creativity_ADHD,
  !is.na(ocir) &
    diva_group %in% c("TD", "ADHD")
)

ocir_data$diva_group <- factor(
  ocir_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
ocir_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_ocir <- ocir_data$ocir[
      ocir_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_ocir),
      Mean = mean(group_ocir),
      SD = sd(group_ocir)
    )
  })
)

ocir_descriptives$Mean <- round(
  ocir_descriptives$Mean,
  2
)

ocir_descriptives$SD <- round(
  ocir_descriptives$SD,
  2
)

print(ocir_descriptives)

# ---- Bayesian analysis of OCI-R ----

# Standardize OCI-R for the Bayesian model
ocir_center <- mean(ocir_data$ocir)
ocir_scale <- sd(ocir_data$ocir)

ocir_data$ocir_z <- (
  ocir_data$ocir - ocir_center
) / ocir_scale

# Fit and save the Bayesian model
ocir_model <- brm(
  formula = bf(
    ocir_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = ocir_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260818,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/ocir_model",
  file_refit = "on_change"
)

# Check model convergence
print(ocir_model)

# Extract posterior draws
ocir_draws <- posterior::as_draws_df(
  ocir_model
)

# Posterior group means on the original OCI-R scale
ocir_mean_TD <- (
  ocir_draws$b_diva_groupTD *
    ocir_scale
) + ocir_center

ocir_mean_ADHD <- (
  ocir_draws$b_diva_groupADHD *
    ocir_scale
) + ocir_center

# Posterior mean difference: ADHD minus TD
ocir_mean_difference <- (
  ocir_mean_ADHD - ocir_mean_TD
)

# Posterior group standard deviations
ocir_sd_TD <- exp(
  ocir_draws$b_sigma_diva_groupTD
) * ocir_scale

ocir_sd_ADHD <- exp(
  ocir_draws$b_sigma_diva_groupADHD
) * ocir_scale

# Group sample sizes
n_TD_ocir <- sum(
  ocir_data$diva_group == "TD"
)

n_ADHD_ocir <- sum(
  ocir_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
ocir_pooled_sd <- sqrt(
  (
    (n_TD_ocir - 1) * ocir_sd_TD^2 +
      (n_ADHD_ocir - 1) * ocir_sd_ADHD^2
  ) /
    (n_TD_ocir + n_ADHD_ocir - 2)
)

# Posterior Cohen's d: ADHD minus TD
ocir_cohens_d <- (
  ocir_mean_difference /
    ocir_pooled_sd
)

# Summarize posterior results
ocir_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      ocir_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      ocir_cohens_d
    )
)

print(
  round(ocir_bayesian_results, 3)
)

# ---- ICAR-16 by group ----

icar_data <- subset(
  df_creativity_ADHD,
  !is.na(icar) &
    diva_group %in% c("TD", "ADHD")
)

icar_data$diva_group <- factor(
  icar_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
icar_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_icar <- icar_data$icar[
      icar_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_icar),
      Mean = mean(group_icar),
      SD = sd(group_icar)
    )
  })
)

icar_descriptives$Mean <- round(
  icar_descriptives$Mean,
  2
)

icar_descriptives$SD <- round(
  icar_descriptives$SD,
  2
)

print(icar_descriptives)

# ---- Bayesian analysis of ICAR-16 ----

# Standardize ICAR-16 for the Bayesian model
icar_center <- mean(icar_data$icar)
icar_scale <- sd(icar_data$icar)

icar_data$icar_z <- (
  icar_data$icar - icar_center
) / icar_scale

# Fit and save the Bayesian model
icar_model <- brm(
  formula = bf(
    icar_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = icar_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260819,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/icar_model",
  file_refit = "on_change"
)

# Check model convergence
print(icar_model)

# Extract posterior draws
icar_draws <- posterior::as_draws_df(
  icar_model
)

# Posterior group means on the original ICAR-16 scale
icar_mean_TD <- (
  icar_draws$b_diva_groupTD *
    icar_scale
) + icar_center

icar_mean_ADHD <- (
  icar_draws$b_diva_groupADHD *
    icar_scale
) + icar_center

# Posterior mean difference: ADHD minus TD
icar_mean_difference <- (
  icar_mean_ADHD - icar_mean_TD
)

# Posterior group standard deviations
icar_sd_TD <- exp(
  icar_draws$b_sigma_diva_groupTD
) * icar_scale

icar_sd_ADHD <- exp(
  icar_draws$b_sigma_diva_groupADHD
) * icar_scale

# Group sample sizes
n_TD_icar <- sum(
  icar_data$diva_group == "TD"
)

n_ADHD_icar <- sum(
  icar_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
icar_pooled_sd <- sqrt(
  (
    (n_TD_icar - 1) * icar_sd_TD^2 +
      (n_ADHD_icar - 1) * icar_sd_ADHD^2
  ) /
    (n_TD_icar + n_ADHD_icar - 2)
)

# Posterior Cohen's d: ADHD minus TD
icar_cohens_d <- (
  icar_mean_difference /
    icar_pooled_sd
)

# Summarize posterior results
icar_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      icar_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      icar_cohens_d
    )
)

print(
  round(icar_bayesian_results, 3)
)

# ---- AQ by group ----

aq_data <- subset(
  df_creativity_ADHD,
  !is.na(aq_sum) &
    diva_group %in% c("TD", "ADHD")
)

aq_data$diva_group <- factor(
  aq_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
aq_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_aq <- aq_data$aq_sum[
      aq_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_aq),
      Mean = mean(group_aq),
      SD = sd(group_aq)
    )
  })
)

aq_descriptives$Mean <- round(
  aq_descriptives$Mean,
  2
)

aq_descriptives$SD <- round(
  aq_descriptives$SD,
  2
)

print(aq_descriptives)

# ---- Bayesian analysis of AQ ----

# Standardize AQ for the Bayesian model
aq_center <- mean(aq_data$aq_sum)
aq_scale <- sd(aq_data$aq_sum)

aq_data$aq_z <- (
  aq_data$aq_sum - aq_center
) / aq_scale

# Fit and save the Bayesian model
aq_model <- brm(
  formula = bf(
    aq_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = aq_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260820,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/aq_model",
  file_refit = "on_change"
)

# Check model convergence
print(aq_model)

# Extract posterior draws
aq_draws <- posterior::as_draws_df(
  aq_model
)

# Posterior group means on the original AQ scale
aq_mean_TD <- (
  aq_draws$b_diva_groupTD *
    aq_scale
) + aq_center

aq_mean_ADHD <- (
  aq_draws$b_diva_groupADHD *
    aq_scale
) + aq_center

# Posterior mean difference: ADHD minus TD
aq_mean_difference <- (
  aq_mean_ADHD - aq_mean_TD
)

# Posterior group standard deviations
aq_sd_TD <- exp(
  aq_draws$b_sigma_diva_groupTD
) * aq_scale

aq_sd_ADHD <- exp(
  aq_draws$b_sigma_diva_groupADHD
) * aq_scale

# Group sample sizes
n_TD_aq <- sum(
  aq_data$diva_group == "TD"
)

n_ADHD_aq <- sum(
  aq_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
aq_pooled_sd <- sqrt(
  (
    (n_TD_aq - 1) * aq_sd_TD^2 +
      (n_ADHD_aq - 1) * aq_sd_ADHD^2
  ) /
    (n_TD_aq + n_ADHD_aq - 2)
)

# Posterior Cohen's d: ADHD minus TD
aq_cohens_d <- (
  aq_mean_difference /
    aq_pooled_sd
)

# Summarize posterior results
aq_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      aq_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      aq_cohens_d
    )
)

print(
  round(aq_bayesian_results, 3)
)

# ---- PQ-B by group ----

pqb_data <- subset(
  df_creativity_ADHD,
  !is.na(pqb) &
    diva_group %in% c("TD", "ADHD")
)

pqb_data$diva_group <- factor(
  pqb_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
pqb_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_pqb <- pqb_data$pqb[
      pqb_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_pqb),
      Mean = mean(group_pqb),
      SD = sd(group_pqb)
    )
  })
)

pqb_descriptives$Mean <- round(
  pqb_descriptives$Mean,
  2
)

pqb_descriptives$SD <- round(
  pqb_descriptives$SD,
  2
)

print(pqb_descriptives)

# ---- Bayesian analysis of PQ-B ----

# Standardize PQ-B for the Bayesian model
pqb_center <- mean(pqb_data$pqb)
pqb_scale <- sd(pqb_data$pqb)

pqb_data$pqb_z <- (
  pqb_data$pqb - pqb_center
) / pqb_scale

# Fit and save the Bayesian model
pqb_model <- brm(
  formula = bf(
    pqb_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = pqb_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260821,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/pqb_model",
  file_refit = "on_change"
)

# Check model convergence
print(pqb_model)

# Extract posterior draws
pqb_draws <- posterior::as_draws_df(
  pqb_model
)

# Posterior group means on the original PQ-B scale
pqb_mean_TD <- (
  pqb_draws$b_diva_groupTD *
    pqb_scale
) + pqb_center

pqb_mean_ADHD <- (
  pqb_draws$b_diva_groupADHD *
    pqb_scale
) + pqb_center

# Posterior mean difference: ADHD minus TD
pqb_mean_difference <- (
  pqb_mean_ADHD - pqb_mean_TD
)

# Posterior group standard deviations
pqb_sd_TD <- exp(
  pqb_draws$b_sigma_diva_groupTD
) * pqb_scale

pqb_sd_ADHD <- exp(
  pqb_draws$b_sigma_diva_groupADHD
) * pqb_scale

# Group sample sizes
n_TD_pqb <- sum(
  pqb_data$diva_group == "TD"
)

n_ADHD_pqb <- sum(
  pqb_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
pqb_pooled_sd <- sqrt(
  (
    (n_TD_pqb - 1) * pqb_sd_TD^2 +
      (n_ADHD_pqb - 1) * pqb_sd_ADHD^2
  ) /
    (n_TD_pqb + n_ADHD_pqb - 2)
)

# Posterior Cohen's d: ADHD minus TD
pqb_cohens_d <- (
  pqb_mean_difference /
    pqb_pooled_sd
)

# Summarize posterior results
pqb_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      pqb_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      pqb_cohens_d
    )
)

print(
  round(pqb_bayesian_results, 3)
)

# ---- AUDIT by group ----

audit_data <- subset(
  df_creativity_ADHD,
  !is.na(AUDIT) &
    diva_group %in% c("TD", "ADHD")
)

audit_data$diva_group <- factor(
  audit_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
audit_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_audit <- audit_data$AUDIT[
      audit_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_audit),
      Mean = mean(group_audit),
      SD = sd(group_audit)
    )
  })
)

audit_descriptives$Mean <- round(
  audit_descriptives$Mean,
  2
)

audit_descriptives$SD <- round(
  audit_descriptives$SD,
  2
)

print(audit_descriptives)

# ---- Bayesian analysis of AUDIT ----

# Standardize AUDIT for the Bayesian model
audit_center <- mean(audit_data$AUDIT)
audit_scale <- sd(audit_data$AUDIT)

audit_data$audit_z <- (
  audit_data$AUDIT - audit_center
) / audit_scale

# Fit and save the Bayesian model
audit_model <- brm(
  formula = bf(
    audit_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = audit_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260822,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/audit_model",
  file_refit = "on_change"
)

# Check model convergence
print(audit_model)

# Extract posterior draws
audit_draws <- posterior::as_draws_df(
  audit_model
)

# Posterior group means on the original AUDIT scale
audit_mean_TD <- (
  audit_draws$b_diva_groupTD *
    audit_scale
) + audit_center

audit_mean_ADHD <- (
  audit_draws$b_diva_groupADHD *
    audit_scale
) + audit_center

# Posterior mean difference: ADHD minus TD
audit_mean_difference <- (
  audit_mean_ADHD - audit_mean_TD
)

# Posterior group standard deviations
audit_sd_TD <- exp(
  audit_draws$b_sigma_diva_groupTD
) * audit_scale

audit_sd_ADHD <- exp(
  audit_draws$b_sigma_diva_groupADHD
) * audit_scale

# Group sample sizes
n_TD_audit <- sum(
  audit_data$diva_group == "TD"
)

n_ADHD_audit <- sum(
  audit_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
audit_pooled_sd <- sqrt(
  (
    (n_TD_audit - 1) * audit_sd_TD^2 +
      (n_ADHD_audit - 1) * audit_sd_ADHD^2
  ) /
    (n_TD_audit + n_ADHD_audit - 2)
)

# Posterior Cohen's d: ADHD minus TD
audit_cohens_d <- (
  audit_mean_difference /
    audit_pooled_sd
)

# Summarize posterior results
audit_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      audit_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      audit_cohens_d
    )
)

print(
  round(audit_bayesian_results, 3)
)

# ---- CUDIT by group ----

cudit_data <- subset(
  df_creativity_ADHD,
  !is.na(CUDIT) &
    diva_group %in% c("TD", "ADHD")
)

cudit_data$diva_group <- factor(
  cudit_data$diva_group,
  levels = c("TD", "ADHD")
)

# Descriptive statistics
cudit_descriptives <- do.call(
  rbind,
  lapply(c("TD", "ADHD"), function(group_name) {
    
    group_cudit <- cudit_data$CUDIT[
      cudit_data$diva_group == group_name
    ]
    
    data.frame(
      Group = group_name,
      N = length(group_cudit),
      Mean = mean(group_cudit),
      SD = sd(group_cudit)
    )
  })
)

cudit_descriptives$Mean <- round(
  cudit_descriptives$Mean,
  2
)

cudit_descriptives$SD <- round(
  cudit_descriptives$SD,
  2
)

print(cudit_descriptives)


# ---- Bayesian analysis of CUDIT ----

# Standardize CUDIT for the Bayesian model
cudit_center <- mean(cudit_data$CUDIT)
cudit_scale <- sd(cudit_data$CUDIT)

cudit_data$cudit_z <- (
  cudit_data$CUDIT - cudit_center
) / cudit_scale

# Fit and save the Bayesian model
cudit_model <- brm(
  formula = bf(
    cudit_z ~ 0 + diva_group,
    sigma ~ 0 + diva_group
  ),
  data = cudit_data,
  family = gaussian(),
  prior = c(
    prior(
      normal(0, 1),
      class = "b"
    ),
    prior(
      normal(0, 0.5),
      class = "b",
      dpar = "sigma"
    )
  ),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(
    4,
    parallel::detectCores()
  ),
  seed = 20260823,
  control = list(
    adapt_delta = 0.95
  ),
  file = "Analysis/models/cudit_model",
  file_refit = "on_change"
)

# Check model convergence
print(cudit_model)

# Extract posterior draws
cudit_draws <- posterior::as_draws_df(
  cudit_model
)

# Posterior group means on the original CUDIT scale
cudit_mean_TD <- (
  cudit_draws$b_diva_groupTD *
    cudit_scale
) + cudit_center

cudit_mean_ADHD <- (
  cudit_draws$b_diva_groupADHD *
    cudit_scale
) + cudit_center

# Posterior mean difference: ADHD minus TD
cudit_mean_difference <- (
  cudit_mean_ADHD - cudit_mean_TD
)

# Posterior group standard deviations
cudit_sd_TD <- exp(
  cudit_draws$b_sigma_diva_groupTD
) * cudit_scale

cudit_sd_ADHD <- exp(
  cudit_draws$b_sigma_diva_groupADHD
) * cudit_scale

# Group sample sizes
n_TD_cudit <- sum(
  cudit_data$diva_group == "TD"
)

n_ADHD_cudit <- sum(
  cudit_data$diva_group == "ADHD"
)

# Posterior pooled standard deviation
cudit_pooled_sd <- sqrt(
  (
    (n_TD_cudit - 1) * cudit_sd_TD^2 +
      (n_ADHD_cudit - 1) * cudit_sd_ADHD^2
  ) /
    (n_TD_cudit + n_ADHD_cudit - 2)
)

# Posterior Cohen's d: ADHD minus TD
cudit_cohens_d <- (
  cudit_mean_difference /
    cudit_pooled_sd
)

# Summarize posterior results
cudit_bayesian_results <- rbind(
  "Mean difference (ADHD - TD)" =
    posterior_result(
      cudit_mean_difference
    ),
  
  "Cohen's d" =
    posterior_result(
      cudit_cohens_d
    )
)

print(
  round(cudit_bayesian_results, 3)
)