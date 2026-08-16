# 1. Packages----

library(tidyverse)

# We will load the Bayesian packages later when we start
# fitting the models:
# library(brms)
# library(cmdstanr)
# library(posterior)
# library(bayesplot)
# library(bayestestR)


# 2. Load data----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)


# 3. Initial checks----

# Check sample size
nrow(df)

# Check the DIVA diagnostic groups
table(df$diva_group, useNA = "ifany")

# Check that declared group and DIVA group agree
table(
  df$declared_group,
  df$diva_group,
  useNA = "ifany"
)

# Check the ADHD subtype variable we created
table(df$ADHD_subtype, useNA = "ifany")

# Also inspect the original DIVA diagnosis types
table(df$diva_diagnosis_type, useNA = "ifany")

# 4. Define analysis groups ----

df <- df %>%
  mutate(
    
    # --------------------------------------------------------
    # Main comparison: ADHD vs. Without ADHD
    # Based on the DIVA diagnostic classification
    # --------------------------------------------------------
    
    group = case_when(
      diva_group == "TD"   ~ "Without ADHD",
      diva_group == "ADHD" ~ "ADHD",
      TRUE                  ~ NA_character_
    ),
    
    group = factor(
      group,
      levels = c("Without ADHD", "ADHD")
    ),
    
    
    # Contrast coding:
    # Without ADHD = -0.5
    # ADHD         = +0.5
    #
    # This means that in regression models:
    # a positive group coefficient = ADHD > Without ADHD
    # a negative group coefficient = ADHD < Without ADHD
    
    group_c = case_when(
      group == "Without ADHD" ~ -0.5,
      group == "ADHD"         ~  0.5,
      TRUE                    ~ NA_real_
    ),
    
    
    # --------------------------------------------------------
    # Secondary three-group analysis
    # --------------------------------------------------------
    #
    # For now we keep the two primary H/I participants together
    # with the combined group, as we discussed.
    #
    # We call this group ADHD_HI rather than "Combined" because
    # it includes both:
    # - Combined presentation
    # - Primary hyperactive/impulsive presentation
    
    subtype3 = case_when(
      ADHD_subtype == "none"        ~ "Without ADHD",
      ADHD_subtype == "inattentive" ~ "Inattentive",
      ADHD_subtype == "combined"     ~ "ADHD_HI",
      TRUE                           ~ NA_character_
    ),
    
    subtype3 = factor(
      subtype3,
      levels = c(
        "Without ADHD",
        "Inattentive",
        "ADHD_HI"
      )
    )
  )

# 5. Verify the new variables----


table(df$group, useNA = "ifany")
table(df$subtype3, useNA = "ifany")

# Cross-check subtype variable against original DIVA diagnosis
table(
  df$subtype3,
  df$diva_diagnosis_type,
  useNA = "ifany"
)

# 6. Identify the primary CFG variables----

# Show variables that are likely relevant to our analyses
names(df)[
  str_detect(
    names(df),
    regex(
      "galler|orig|eff|optimal|g_empirical|alpha_empirical|median.*step",
      ignore_case = TRUE
    )
  )
]


# 7. Missing values in variables of interest----

vars_main <- c(
  "#galleries",
  "Gallery Orig",
  "g_empirical",
  "alpha_empirical",
  "exp efficiency",
  "scav efficiency",
  "Gallery Orig exp",
  "Gallery Orig scav"
)

df %>%
  summarise(
    across(
      all_of(vars_main),
      ~ sum(is.na(.x))
    )
  )


# 8. Descriptive statistics: ADHD vs Without ADHD ----

descriptives_main <- df %>%
  group_by(group) %>%
  summarise(
    n = n(),
    
    # Fluency
    fluency_mean   = mean(`#galleries`, na.rm = TRUE),
    fluency_sd     = sd(`#galleries`, na.rm = TRUE),
    fluency_median = median(`#galleries`, na.rm = TRUE),
    fluency_q1     = quantile(`#galleries`, 0.25, na.rm = TRUE),
    fluency_q3     = quantile(`#galleries`, 0.75, na.rm = TRUE),
    
    # Overall originality
    originality_mean   = mean(`Gallery Orig`, na.rm = TRUE),
    originality_sd     = sd(`Gallery Orig`, na.rm = TRUE),
    originality_median = median(`Gallery Orig`, na.rm = TRUE),
    originality_q1     = quantile(`Gallery Orig`, 0.25, na.rm = TRUE),
    originality_q3     = quantile(`Gallery Orig`, 0.75, na.rm = TRUE),
    
    # g
    g_mean   = mean(g_empirical, na.rm = TRUE),
    g_sd     = sd(g_empirical, na.rm = TRUE),
    g_median = median(g_empirical, na.rm = TRUE),
    g_q1     = quantile(g_empirical, 0.25, na.rm = TRUE),
    g_q3     = quantile(g_empirical, 0.75, na.rm = TRUE),
    
    # alpha
    alpha_mean   = mean(alpha_empirical, na.rm = TRUE),
    alpha_sd     = sd(alpha_empirical, na.rm = TRUE),
    alpha_median = median(alpha_empirical, na.rm = TRUE),
    alpha_q1     = quantile(alpha_empirical, 0.25, na.rm = TRUE),
    alpha_q3     = quantile(alpha_empirical, 0.75, na.rm = TRUE),
    
    # Exploration efficiency
    exp_eff_mean   = mean(`exp efficiency`, na.rm = TRUE),
    exp_eff_sd     = sd(`exp efficiency`, na.rm = TRUE),
    exp_eff_median = median(`exp efficiency`, na.rm = TRUE),
    exp_eff_q1     = quantile(`exp efficiency`, 0.25, na.rm = TRUE),
    exp_eff_q3     = quantile(`exp efficiency`, 0.75, na.rm = TRUE),
    
    # Exploitation efficiency
    scav_eff_mean   = mean(`scav efficiency`, na.rm = TRUE),
    scav_eff_sd     = sd(`scav efficiency`, na.rm = TRUE),
    scav_eff_median = median(`scav efficiency`, na.rm = TRUE),
    scav_eff_q1     = quantile(`scav efficiency`, 0.25, na.rm = TRUE),
    scav_eff_q3     = quantile(`scav efficiency`, 0.75, na.rm = TRUE)
  )

descriptives_main


# 9. Originality descriptives by search phase ----

descriptives_originality_phase <- df %>%
  group_by(group) %>%
  summarise(
    
    # Overall originality
    overall_mean   = mean(`Gallery Orig`, na.rm = TRUE),
    overall_sd     = sd(`Gallery Orig`, na.rm = TRUE),
    overall_median = median(`Gallery Orig`, na.rm = TRUE),
    
    # Exploration originality
    exploration_mean   = mean(`Gallery Orig exp`, na.rm = TRUE),
    exploration_sd     = sd(`Gallery Orig exp`, na.rm = TRUE),
    exploration_median = median(`Gallery Orig exp`, na.rm = TRUE),
    
    # Exploitation originality
    exploitation_mean   = mean(`Gallery Orig scav`, na.rm = TRUE),
    exploitation_sd     = sd(`Gallery Orig scav`, na.rm = TRUE),
    exploitation_median = median(`Gallery Orig scav`, na.rm = TRUE)
  )

descriptives_originality_phase

# 10. Bayesian packages and setup ----

library(brms)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(bayestestR)

set.seed(2026)

options(mc.cores = parallel::detectCores())

### ORIGINALLITY ####
# 11. Prepare overall originality outcome ----

# Save the original mean and SD so we can later transform
# posterior estimates back to the original originality scale
orig_center <- mean(df$`Gallery Orig`, na.rm = TRUE)
orig_scale  <- sd(df$`Gallery Orig`, na.rm = TRUE)

df <- df %>%
  mutate(
    originality_z = (`Gallery Orig` - orig_center) / orig_scale
  )

# 12. Priors for continuous outcomes ----

priors_continuous <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)

# 13. Overall originality: ADHD vs Without ADHD ----

m_orig <- brm(
  originality_z ~ group_c,
  data = df,
  family = student(),
  
  prior = priors_continuous,
  
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

summary(m_orig)

# Posterior summary of the ADHD - Without ADHD difference

orig_posterior <- describe_posterior(
  m_orig,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

orig_posterior
# 14. Posterior draws and back-transformation ----

# Original scale parameters
orig_center <- mean(df$`Gallery Orig`, na.rm = TRUE)
orig_scale  <- sd(df$`Gallery Orig`, na.rm = TRUE)

# Extract all posterior draws from the model
orig_draws <- as_draws_df(m_orig)

# Calculate posterior estimates for each group
# and for the ADHD - Without ADHD difference
orig_draws <- orig_draws %>%
  mutate(
    
    # Estimated group locations on standardized scale
    without_adhd_z = b_Intercept - 0.5 * b_group_c,
    adhd_z         = b_Intercept + 0.5 * b_group_c,
    
    # Transform estimates back to original originality scale
    without_adhd_orig = without_adhd_z * orig_scale + orig_center,
    adhd_orig         = adhd_z * orig_scale + orig_center,
    
    # ADHD - Without ADHD difference
    diff_z    = b_group_c,
    diff_orig = b_group_c * orig_scale
  )

head(orig_draws)

# Posterior ADHD - Without ADHD difference
# in the original originality scale

orig_difference_original_scale <- describe_posterior(
  orig_draws$diff_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

orig_difference_original_scale


# Posterior estimates for each group
# in the original originality scale

orig_group_summary <- tibble(
  group = c("Without ADHD", "ADHD"),
  
  posterior_median = c(
    median(orig_draws$without_adhd_orig),
    median(orig_draws$adhd_orig)
  ),
  
  lower_90 = c(
    quantile(orig_draws$without_adhd_orig, 0.05),
    quantile(orig_draws$adhd_orig, 0.05)
  ),
  
  upper_90 = c(
    quantile(orig_draws$without_adhd_orig, 0.95),
    quantile(orig_draws$adhd_orig, 0.95)
  )
)

orig_group_summary

# 15. Figure data for overall originality ----

# Summary of posterior difference (ADHD - without ADHD) in original units
orig_diff_summary <- tibble(
  median   = median(orig_draws$diff_orig),
  lower_90 = quantile(orig_draws$diff_orig, 0.05),
  upper_90 = quantile(orig_draws$diff_orig, 0.95)
)

orig_diff_summary

# 16. Plot A: raw observed originality by group ----

p_orig_raw <- ggplot(
  df,
  aes(x = group, y = `Gallery Orig`, color = group)
) +
  geom_jitter(
    width = 0.12,
    alpha = 0.30,
    size = 2,
    show.legend = FALSE
  ) +

  # Mean ± 1 SD
  stat_summary(
    fun.data = mean_sdl,
    fun.args = list(mult = 1),
    geom = "errorbar",
    width = 0.05,
    linewidth = 0.6,
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  # Group mean
  stat_summary(
    fun = mean,
    geom = "point",
    size = 3.5,
    color = "black",
    show.legend = FALSE
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  labs(
    x = NULL,
    y = "Overall originality",
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    axis.text = element_text(size = 13),
    axis.title = element_text(size = 15)
  )

p_orig_raw

# Plot B: posterior distribution difference of ADHD - without ADHD ----

p_orig_diff <- ggplot(
  orig_draws,
  aes(x = diff_orig)
) +
  
  # Posterior distribution of the group difference
  geom_density(
    fill = "#808080",
    alpha = 0.40,
    color = "#555555",
    linewidth = 1
  ) +
  
  # Zero = no group difference
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  # Posterior median
  annotate(
    "point",
    x = orig_diff_summary$median,
    y = 0,
    size = 3,
    color = "black"
  ) +
  
  labs(
    x = "Estimated ADHD - Without ADHD difference in overall originality",
    y = NULL,
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    # Remove y-axis
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "none",
    
    # Flatter figure, similar to previous paper
    aspect.ratio = 0.45
  )

p_orig_diff


# 18. Plot C: posterior distributions for each group ----

library(tidyr)

orig_groups_long <- orig_draws %>%
  select(without_adhd_orig, adhd_orig) %>%
  rename(
    `Without ADHD` = without_adhd_orig,
    ADHD = adhd_orig
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "originality"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c("ADHD", "Without ADHD")
    )
  )

group_medians <- orig_groups_long %>%
  group_by(group) %>%
  summarise(
    median = median(originality),
    .groups = "drop"
  )

p_orig_groups <- ggplot(
  orig_groups_long,
  aes(
    x = originality,
    fill = group,
    color = group
  )
) +
  
  # Posterior distributions
  geom_density(
    alpha = 0.65,
    linewidth = 0.8,
    adjust = 1
  ) +
  
  # Grey baseline, similar to Panel C
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  # Black points = posterior medians
  geom_point(
    data = group_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  labs(
    x = "Estimated overall originality",
    y = NULL,
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    # Panel label
    plot.title = element_text(
      face = "bold",
      size = 18
    ),
    
    # Remove y axis completely
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    # X-axis formatting
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    # legend
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    # Make the figure flatter
    aspect.ratio = 0.45
  )

p_orig_groups

# 19. Save plots ----

ggsave(
  filename = "Figures/originality/originality_raw_by_group.png",
  plot = p_orig_raw,
  width = 6,
  height = 5,
  dpi = 300
)

ggsave(
  filename = "Figures/originality/originality_posterior_difference.png",
  plot = p_orig_diff,
  width = 6,
  height = 5,
  dpi = 300
)

ggsave(
  filename = "Figures/originality/originality_posterior_groups.png",
  plot = p_orig_groups,
  width = 6,
  height = 4,
  dpi = 300
)

# 20. PHASE-SPECIFIC ORIGINALITY: Exploration vs Exploitation ----

## 20.1 Prepare phase-specific originality data ----

orig_phase <- df %>%
  select(
    ID,
    group,
    group_c,
    `Gallery Orig exp`,
    `Gallery Orig scav`
  ) %>%
  
  pivot_longer(
    cols = c(
      `Gallery Orig exp`,
      `Gallery Orig scav`
    ),
    names_to = "phase",
    values_to = "originality"
  ) %>%
  
  mutate(
    
    # Give the phases clear names
    phase = case_when(
      phase == "Gallery Orig exp"  ~ "Exploration",
      phase == "Gallery Orig scav" ~ "Exploitation"
    ),
    
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    ),
    
    # Contrast coding:
    # Exploration  = -0.5
    # Exploitation = +0.5
    phase_c = case_when(
      phase == "Exploration"  ~ -0.5,
      phase == "Exploitation" ~  0.5
    )
  )

## 20.3 Standardize phase-specific originality ----

orig_phase_center <- mean(
  orig_phase$originality,
  na.rm = TRUE
)

orig_phase_scale <- sd(
  orig_phase$originality,
  na.rm = TRUE
)

orig_phase <- orig_phase %>%
  mutate(
    originality_z =
      (originality - orig_phase_center) / orig_phase_scale
  )

## 20.4 Priors for phase-specific originality model ----

priors_orig_phase <- c(
  
  # Fixed effects:
  # Group, Phase, and Group x Phase
  prior(normal(0, 0.5), class = "b"),
  
  prior(normal(0, 1), class = "Intercept"),
  
  # Residual variability
  prior(exponential(1), class = "sigma"),
  
  # Between-participant variability
  prior(exponential(1), class = "sd"),
  
  # Degrees of freedom for Student-t likelihood
  prior(gamma(2, 0.1), class = "nu")
)

## 20.5 Bayesian Group x Phase model ----

m_orig_phase <- brm(
  
  originality_z ~ group_c * phase_c + (1 | ID),
  
  data = orig_phase,
  family = student(),
  
  prior = priors_orig_phase,
  
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

summary(m_orig_phase)

## 20.6 Posterior summary ----

orig_phase_posterior <- describe_posterior(
  m_orig_phase,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

orig_phase_posterior

## 20.7 ADHD - Without ADHD difference within each phase ----

# Extract posterior draws
orig_phase_draws <- as_draws_df(m_orig_phase)

# Calculate the ADHD - Without ADHD difference
# separately for Exploration and Exploitation
orig_phase_draws <- orig_phase_draws %>%
  mutate(
    
    # Exploration is coded -0.5
    diff_exploration_z =
      b_group_c - 0.5 * `b_group_c:phase_c`,
    
    # Exploitation is coded +0.5
    diff_exploitation_z =
      b_group_c + 0.5 * `b_group_c:phase_c`,
    
    # Transform differences back to original originality units
    diff_exploration_orig =
      diff_exploration_z * orig_phase_scale,
    
    diff_exploitation_orig =
      diff_exploitation_z * orig_phase_scale
  )

## 20.8 Posterior summaries of group differences by phase ----

# Exploration
orig_exploration_difference <- describe_posterior(
  orig_phase_draws$diff_exploration_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

# Exploitation
orig_exploitation_difference <- describe_posterior(
  orig_phase_draws$diff_exploitation_orig,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

orig_exploration_difference
orig_exploitation_difference

## 20.9 Standardized group differences by phase ----

orig_exploration_difference_z <- describe_posterior(
  orig_phase_draws$diff_exploration_z,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

orig_exploitation_difference_z <- describe_posterior(
  orig_phase_draws$diff_exploitation_z,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)


orig_exploration_difference_z
orig_exploitation_difference_z

# 20.10 Plot posterior group differences by search phase ----

## 20.10 Prepare posterior draws for phase-difference plot ----

orig_phase_diff_long <- orig_phase_draws %>%
  select(
    diff_exploration_orig,
    diff_exploitation_orig
  ) %>%
  rename(
    Exploration = diff_exploration_orig,
    Exploitation = diff_exploitation_orig
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "phase",
    values_to = "group_difference"
  ) %>%
  mutate(
    phase = factor(
      phase,
      levels = c("Exploration", "Exploitation")
    )
  )

## 20.11 Posterior medians for plotting ----

orig_phase_diff_medians <- orig_phase_diff_long %>%
  group_by(phase) %>%
  summarise(
    median = median(group_difference),
    .groups = "drop"
  )

orig_phase_diff_medians

## 20.12 Plot: ADHD - Without ADHD difference by search phase ----

p_orig_phase_diff <- ggplot(
  orig_phase_diff_long,
  aes(
    x = group_difference,
    fill = phase,
    color = phase
  )
) +
  
  
  
  # Posterior distributions
  geom_density(
    alpha = 0.45,
    linewidth = 1,
    adjust = 1
  ) +
  
  # Zero = no difference between groups
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  # Posterior medians
  geom_point(
    data = orig_phase_diff_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  # Phase colors
  scale_fill_manual(
    name = NULL,
    values = c(
      "Exploration" = "#E69F00",
      "Exploitation" = "#009E73"
    )
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "Exploration" = "#E69F00",
      "Exploitation" = "#009E73"
    )
  ) +
  
  
  labs(
    x = "Estimated ADHD - Without ADHD difference in originality",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    panel.background = element_rect(
      fill = "white",
      color = NA
    ),
    plot.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.grid = element_blank(),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_orig_phase_diff



# 20.13 Save phase-specific originality plot ----

ggsave(
  filename = "Figures/originality/originality_group_difference_by_phase.png",
  plot = p_orig_phase_diff,
  width = 9,
  height = 4.5,
  dpi = 300
)

# 21. POSTERIOR GROUP DISTRIBUTIONS WITHIN EACH PHASE ----

## 21.1 Calculate posterior estimates for each group in each phase ----

phase_group_draws <- orig_phase_draws %>%
  transmute(
    
    # Exploration: phase_c = -0.5
    
    without_adhd_exploration_z =
      b_Intercept -
      0.5 * b_group_c -
      0.5 * b_phase_c +
      0.25 * `b_group_c:phase_c`,
    
    adhd_exploration_z =
      b_Intercept +
      0.5 * b_group_c -
      0.5 * b_phase_c -
      0.25 * `b_group_c:phase_c`,
    
    
    # Exploitation: phase_c = +0.5
    
    without_adhd_exploitation_z =
      b_Intercept -
      0.5 * b_group_c +
      0.5 * b_phase_c -
      0.25 * `b_group_c:phase_c`,
    
    adhd_exploitation_z =
      b_Intercept +
      0.5 * b_group_c +
      0.5 * b_phase_c +
      0.25 * `b_group_c:phase_c`
  ) %>%
  
  mutate(
    
    # Back-transform to original originality units
    
    without_adhd_exploration =
      without_adhd_exploration_z * orig_phase_scale + orig_phase_center,
    
    adhd_exploration =
      adhd_exploration_z * orig_phase_scale + orig_phase_center,
    
    without_adhd_exploitation =
      without_adhd_exploitation_z * orig_phase_scale + orig_phase_center,
    
    adhd_exploitation =
      adhd_exploitation_z * orig_phase_scale + orig_phase_center
  )

## 21.2 Posterior summaries for each group within each phase ----

phase_group_summary <- tibble(
  
  group = c(
    "Without ADHD",
    "ADHD",
    "Without ADHD",
    "ADHD"
  ),
  
  phase = c(
    "Exploration",
    "Exploration",
    "Exploitation",
    "Exploitation"
  ),
  
  posterior_median = c(
    median(phase_group_draws$without_adhd_exploration),
    median(phase_group_draws$adhd_exploration),
    median(phase_group_draws$without_adhd_exploitation),
    median(phase_group_draws$adhd_exploitation)
  ),
  
  lower_90 = c(
    quantile(phase_group_draws$without_adhd_exploration, 0.05),
    quantile(phase_group_draws$adhd_exploration, 0.05),
    quantile(phase_group_draws$without_adhd_exploitation, 0.05),
    quantile(phase_group_draws$adhd_exploitation, 0.05)
  ),
  
  upper_90 = c(
    quantile(phase_group_draws$without_adhd_exploration, 0.95),
    quantile(phase_group_draws$adhd_exploration, 0.95),
    quantile(phase_group_draws$without_adhd_exploitation, 0.95),
    quantile(phase_group_draws$adhd_exploitation, 0.95)
  )
)

phase_group_summary

## 21.3 Prepare plotting data ----

# Exploration
orig_exploration_groups <- phase_group_draws %>%
  select(
    adhd_exploration,
    without_adhd_exploration
  ) %>%
  rename(
    ADHD = adhd_exploration,
    `Without ADHD` = without_adhd_exploration
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "originality"
  )

exploration_group_medians <- orig_exploration_groups %>%
  group_by(group) %>%
  summarise(
    median = median(originality),
    .groups = "drop"
  )


# Exploitation
orig_exploitation_groups <- phase_group_draws %>%
  select(
    adhd_exploitation,
    without_adhd_exploitation
  ) %>%
  rename(
    ADHD = adhd_exploitation,
    `Without ADHD` = without_adhd_exploitation
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "originality"
  )

exploitation_group_medians <- orig_exploitation_groups %>%
  group_by(group) %>%
  summarise(
    median = median(originality),
    .groups = "drop"
  )

## 21.4 Common x-axis limits for both phase plots ----

phase_group_x_limits <- range(
  c(
    orig_exploration_groups$originality,
    orig_exploitation_groups$originality
  ),
  na.rm = TRUE
)

phase_group_x_limits

## 21.5 Plot: Exploration originality by group ----

p_orig_exploration_groups <- ggplot(
  orig_exploration_groups,
  aes(
    x = originality,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.45,
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = exploration_group_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  labs(
    x = "Estimated exploration originality",
    y = NULL
  ) +
  
  coord_cartesian(
    xlim = phase_group_x_limits
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_orig_exploration_groups

## 21.6 Plot: Exploitation originality by group ----

p_orig_exploitation_groups <- ggplot(
  orig_exploitation_groups,
  aes(
    x = originality,
    fill = group,
    color = group
  )
) +
  
  geom_density(
    alpha = 0.45,
    linewidth = 1
  ) +
  
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = exploitation_group_medians,
    aes(
      x = median,
      y = 0
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "ADHD" = "#0072B2",
      "Without ADHD" = "#CC79A7"
    )
  ) +
  
  labs(
    x = "Estimated exploitation originality",
    y = NULL
  ) +
  
  coord_cartesian(
    xlim = phase_group_x_limits
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_orig_exploitation_groups

## 21.7 Save phase-specific group posterior plots ----

ggsave(
  filename = "Figures/originality/originality_exploration_groups.png",
  plot = p_orig_exploration_groups,
  width = 8,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = "Figures/originality/originality_exploitation_groups.png",
  plot = p_orig_exploitation_groups,
  width = 8,
  height = 4.5,
  dpi = 300,
  bg = "white"
)