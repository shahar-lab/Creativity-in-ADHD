# ============================================================
# SEARCH DYNAMICS: ALPHA AND G
# ADHD vs Without ADHD
# ============================================================

# 1. Packages ----

library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(bayestestR)

set.seed(2026)

options(mc.cores = parallel::detectCores())


# 2. Load data ----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)


# 3. Define analysis groups ----

df <- df %>%
  mutate(
    
    group = case_when(
      diva_group == "TD"   ~ "Without ADHD",
      diva_group == "ADHD" ~ "ADHD",
      TRUE                 ~ NA_character_
    ),
    
    group = factor(
      group,
      levels = c("Without ADHD", "ADHD")
    ),
    
    # Contrast coding:
    # Without ADHD = -0.5
    # ADHD         = +0.5
    group_c = case_when(
      group == "Without ADHD" ~ -0.5,
      group == "ADHD"         ~  0.5,
      TRUE                    ~ NA_real_
    )
  )

table(df$group, useNA = "ifany")

-----------------------------------------------------
# 4. ALPHA: Relative exploration-exploitation balance 
-----------------------------------------------------
  
# 4.1 Alpha descriptives by group ----

alpha_descriptives <- df %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(alpha_empirical, na.rm = TRUE),
    sd = sd(alpha_empirical, na.rm = TRUE),
    median = median(alpha_empirical, na.rm = TRUE),
    q1 = quantile(alpha_empirical, 0.25, na.rm = TRUE),
    q3 = quantile(alpha_empirical, 0.75, na.rm = TRUE),
    min = min(alpha_empirical, na.rm = TRUE),
    max = max(alpha_empirical, na.rm = TRUE),
    .groups = "drop"
  )

alpha_descriptives

# 4.2 Standardize alpha ----

alpha_center <- mean(df$alpha_empirical, na.rm = TRUE)
alpha_scale  <- sd(df$alpha_empirical, na.rm = TRUE)

df <- df %>%
  mutate(
    alpha_z = (alpha_empirical - alpha_center) / alpha_scale
  )

# 4.3 Priors for continuous outcomes ----

priors_continuous <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(normal(0, 1), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)

# 4.4 Bayesian alpha model: ADHD vs Without ADHD ----

m_alpha <- brm(
  alpha_z ~ group_c,
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

summary(m_alpha)

# 4.5 Posterior summary ----

alpha_posterior <- describe_posterior(
  m_alpha,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

alpha_posterior

# 4.6 Posterior draws for alpha ----

alpha_draws <- as_draws_df(m_alpha) %>%
  mutate(
    
    # Estimated group locations on standardized alpha scale
    without_adhd_alpha_z = b_Intercept - 0.5 * b_group_c,
    adhd_alpha_z         = b_Intercept + 0.5 * b_group_c,
    
    # ADHD - Without ADHD difference
    diff_alpha_z = b_group_c
  )

# 4.7 Posterior estimates for each group ----

alpha_group_summary <- tibble(
  group = c("Without ADHD", "ADHD"),
  
  posterior_median = c(
    median(alpha_draws$without_adhd_alpha_z),
    median(alpha_draws$adhd_alpha_z)
  ),
  
  lower_90 = c(
    quantile(alpha_draws$without_adhd_alpha_z, 0.05),
    quantile(alpha_draws$adhd_alpha_z, 0.05)
  ),
  
  upper_90 = c(
    quantile(alpha_draws$without_adhd_alpha_z, 0.95),
    quantile(alpha_draws$adhd_alpha_z, 0.95)
  )
)

alpha_group_summary

# 4.8 Plot A: raw observed alpha by group ----

p_alpha_raw <- ggplot(
  df,
  aes(
    x = group,
    y = alpha_empirical,
    color = group
  )
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
    y = expression(alpha)
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text = element_text(size = 13),
    axis.title = element_text(size = 15)
  )

p_alpha_raw

# 4.9 Plot A: posterior alpha distributions by group ----

alpha_groups_long <- alpha_draws %>%
  select(
    without_adhd_alpha_z,
    adhd_alpha_z
  ) %>%
  rename(
    `Without ADHD` = without_adhd_alpha_z,
    ADHD = adhd_alpha_z
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "alpha"
  )

alpha_group_medians <- alpha_groups_long %>%
  group_by(group) %>%
  summarise(
    median = median(alpha),
    .groups = "drop"
  )

p_alpha_groups <- ggplot(
  alpha_groups_long,
  aes(
    x = alpha,
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
    data = alpha_group_medians,
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
    x = "Estimated α score (standardized)",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    legend.position = "right",
    legend.text = element_text(size = 13),
    
    aspect.ratio = 0.45
  )

p_alpha_groups

# 4.10 Plot B: posterior ADHD - Without ADHD alpha difference ----

alpha_diff_median <- median(
  alpha_draws$diff_alpha_z
)

p_alpha_diff <- ggplot(
  alpha_draws,
  aes(x = diff_alpha_z)
) +
  
  geom_density(
    fill = "grey75",
    color = "grey35",
    alpha = 0.8,
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
    x = alpha_diff_median,
    y = 0,
    color = "black",
    size = 3
  ) +
  
  labs(
    x = "Estimated ADHD - Without ADHD difference in α (SD units)",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    aspect.ratio = 0.45
  )

p_alpha_diff

# 4.11 Save alpha plots ----

ggsave(
  filename = "Figures/alpha/alpha_posterior_groups.png",
  plot = p_alpha_groups,
  width = 8,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = "Figures/alpha/alpha_group_difference.png",
  plot = p_alpha_diff,
  width = 7,
  height = 3.5,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = "Figures/alpha/alpha_raw_by_group.png",
  plot = p_alpha_raw,
  width = 6,
  height = 5,
  dpi = 300,
  bg = "white"
)
# ============================================================
# 5. G parameter: Exploration–exploitation cycle duration
# ============================================================

# 5.1 g descriptives by group ----

g_descriptives <- df %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(g_empirical, na.rm = TRUE),
    sd = sd(g_empirical, na.rm = TRUE),
    median = median(g_empirical, na.rm = TRUE),
    q1 = quantile(g_empirical, 0.25, na.rm = TRUE),
    q3 = quantile(g_empirical, 0.75, na.rm = TRUE),
    min = min(g_empirical, na.rm = TRUE),
    max = max(g_empirical, na.rm = TRUE),
    .groups = "drop"
  )

g_descriptives

# 5.2 Standardize g ----

g_center <- mean(df$g_empirical, na.rm = TRUE)
g_scale  <- sd(df$g_empirical, na.rm = TRUE)

df <- df %>%
  mutate(
    g_z = (g_empirical - g_center) / g_scale
  )

# 5.3 Bayesian g model: ADHD vs Without ADHD ----

m_g <- brm(
  g_z ~ group_c,
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

summary(m_g)

# 5.4 Posterior summary ----

g_posterior <- describe_posterior(
  m_g,
  effects = "fixed",
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

g_posterior

# 5.5 Plot A: raw observed g by group ----

p_g_raw <- ggplot(
  df,
  aes(
    x = group,
    y = g_empirical,
    color = group
  )
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
    y = expression(italic(g))
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text = element_text(size = 13),
    axis.title = element_text(size = 15)
  )

p_g_raw

# 5.6 Save raw g plot ----

ggsave(
  filename = "Figures/g/g_raw_by_group.png",
  plot = p_g_raw,
  width = 6.5,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

# 5.7 Posterior draws for g ----

g_draws <- as_draws_df(m_g) %>%
  mutate(
    without_adhd_g_z = b_Intercept - 0.5 * b_group_c,
    adhd_g_z         = b_Intercept + 0.5 * b_group_c,
    diff_g_z         = b_group_c
  )

# 5.8 Posterior estimates for each group ----

g_group_summary <- tibble(
  group = c("Without ADHD", "ADHD"),
  
  posterior_median = c(
    median(g_draws$without_adhd_g_z),
    median(g_draws$adhd_g_z)
  ),
  
  lower_90 = c(
    quantile(g_draws$without_adhd_g_z, 0.05),
    quantile(g_draws$adhd_g_z, 0.05)
  ),
  
  upper_90 = c(
    quantile(g_draws$without_adhd_g_z, 0.95),
    quantile(g_draws$adhd_g_z, 0.95)
  )
)

g_group_summary

# 5.9 Plot B: posterior g distributions by group ----

g_groups_long <- g_draws %>%
  select(
    without_adhd_g_z,
    adhd_g_z
  ) %>%
  rename(
    `Without ADHD` = without_adhd_g_z,
    ADHD = adhd_g_z
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "g_value"
  )

g_group_medians <- g_groups_long %>%
  group_by(group) %>%
  summarise(
    median = median(g_value),
    .groups = "drop"
  )

p_g_groups <- ggplot(
  g_groups_long,
  aes(
    x = g_value,
    fill = group,
    color = group
  )
) +
  geom_density(
    alpha = 0.45,
    linewidth = 1
  ) +
  geom_point(
    data = g_group_medians,
    aes(x = median, y = 0),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  scale_fill_manual(
    name = NULL,
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  scale_color_manual(
    name = NULL,
    values = c(
      "Without ADHD" = "#CC79A7",
      "ADHD" = "#0072B2"
    )
  ) +
  labs(
    x = "Estimated g score (standardized)",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    legend.position = "right",
    legend.text = element_text(size = 13),
    aspect.ratio = 0.45
  )

p_g_groups

# 5.10 Plot C: posterior ADHD - Without ADHD difference in g ----

g_diff_median <- median(g_draws$diff_g_z)

p_g_diff <- ggplot(
  g_draws,
  aes(x = diff_g_z)
) +
  geom_density(
    fill = "grey75",
    color = "grey35",
    alpha = 0.8,
    linewidth = 1
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  annotate(
    "point",
    x = g_diff_median,
    y = 0,
    color = "black",
    size = 3
  ) +
  labs(
    x = "Estimated ADHD - Without ADHD difference in g (SD units)",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    aspect.ratio = 0.45
  )

p_g_diff

# 5.11 Save g plots ----


ggsave(
  filename = "Figures/g/g_posterior_groups.png",
  plot = p_g_groups,
  width = 8,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = "Figures/g/g_group_difference.png",
  plot = p_g_diff,
  width = 7,
  height = 3.5,
  dpi = 300,
  bg = "white"
)

# ADHD PRESENTATION ANALYSIS: g ----


# 1. Define presentation labels ----

df <- df %>%
  mutate(
    presentation3 = factor(
      ADHD_subtype,
      levels = c(
        "none",
        "inattentive",
        "combined"
      ),
      labels = c(
        "Without ADHD",
        "Inattentive",
        "Combined/HI"
      )
    )
  )

table(df$presentation3, useNA = "ifany")

# 2. Standardize g ----

g_center <- mean(
  df$g_empirical,
  na.rm = TRUE
)

g_scale <- sd(
  df$g_empirical,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    g_z = (g_empirical - g_center) / g_scale
  )

# 3. Bayesian model: g across ADHD presentations ----

m_g_presentation <- brm(
  g_z ~ presentation3,
  
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

summary(m_g_presentation)

# 4. Posterior estimates for each presentation group ----

newdata_g_presentation <- tibble(
  presentation3 = factor(
    c(
      "Without ADHD",
      "Inattentive",
      "Combined/HI"
    ),
    levels = levels(df$presentation3)
  )
)

g_presentation_epred <- posterior_epred(
  m_g_presentation,
  newdata = newdata_g_presentation
)


# 5. Posterior summary by presentation group ----

g_presentation_summary <- tibble(
  group = c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  ),
  
  posterior_median = apply(
    g_presentation_epred, 2, median
  ),
  
  lower_90 = apply(
    g_presentation_epred, 2, quantile, probs = 0.05
  ),
  
  upper_90 = apply(
    g_presentation_epred, 2, quantile, probs = 0.95
  )
)

g_presentation_summary

# 6. Pairwise posterior contrasts for interpretation ----

g_presentation_contrasts <- tibble(
  
  contrast = c(
    "Inattentive - Without ADHD",
    "Combined/HI - Without ADHD",
    "Combined/HI - Inattentive"
  ),
  
  median = c(
    median(
      g_presentation_epred[, 2] -
        g_presentation_epred[, 1]
    ),
    
    median(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 1]
    ),
    
    median(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 2]
    )
  ),
  
  lower_90 = c(
    quantile(
      g_presentation_epred[, 2] -
        g_presentation_epred[, 1], 0.05
    ),
    
    quantile(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 1], 0.05
    ),
    
    quantile(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 2], 0.05
    )
  ),
  
  upper_90 = c(
    quantile(
      g_presentation_epred[, 2] -
        g_presentation_epred[, 1], 0.95
    ),
    
    quantile(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 1], 0.95
    ),
    
    quantile(
      g_presentation_epred[, 3] -
        g_presentation_epred[, 2], 0.95
    )
  )
)

g_presentation_contrasts

# 7. Plot: posterior g distributions by ADHD presentation ----

library(tidyr)

g_presentation_long <- as_tibble(g_presentation_epred) %>%
  setNames(c("Without ADHD", "Inattentive", "Combined/HI")) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "g_z"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c("Without ADHD", "Inattentive", "Combined/HI")
    )
  )

g_presentation_medians <- g_presentation_long %>%
  group_by(group) %>%
  summarise(
    median = median(g_z),
    .groups = "drop"
  )

p_g_presentation <- ggplot(
  g_presentation_long,
  aes(x = g_z, fill = group, color = group)
) +
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_point(
    data = g_presentation_medians,
    aes(x = median, y = 0),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  labs(
    x = "Estimated standardized g",
    y = NULL,
    fill = NULL,
    color = NULL
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

p_g_presentation

# 8. Save g presentation plot ----

ggsave(
  filename = "Figures/g/g_posterior_by_ADHD_presentation.png",
  plot = p_g_presentation,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

# Add pd to g presentation contrasts ----

g_diff_inatt_vs_without <-
  g_presentation_epred[, 2] - g_presentation_epred[, 1]

g_diff_combined_vs_without <-
  g_presentation_epred[, 3] - g_presentation_epred[, 1]

g_diff_combined_vs_inatt <-
  g_presentation_epred[, 3] - g_presentation_epred[, 2]


g_presentation_pd <- tibble(
  contrast = c(
    "Inattentive - Without ADHD",
    "Combined/HI - Without ADHD",
    "Combined/HI - Inattentive"
  ),
  
  pd = c(
    max(
      mean(g_diff_inatt_vs_without > 0),
      mean(g_diff_inatt_vs_without < 0)
    ),
    
    max(
      mean(g_diff_combined_vs_without > 0),
      mean(g_diff_combined_vs_without < 0)
    ),
    
    max(
      mean(g_diff_combined_vs_inatt > 0),
      mean(g_diff_combined_vs_inatt < 0)
    )
  ) * 100
)

g_presentation_pd

# ADHD PRESENTATION ANALYSIS: ALPHA ----


# 1. Define presentation labels ----

df <- df %>%
  mutate(
    presentation3 = factor(
      ADHD_subtype,
      levels = c(
        "none",
        "inattentive",
        "combined"
      ),
      labels = c(
        "Without ADHD",
        "Inattentive",
        "Combined/HI"
      )
    )
  )

table(df$presentation3, useNA = "ifany")

# 2. Standardize alpha ----

alpha_center <- mean(
  df$alpha_empirical,
  na.rm = TRUE
)

alpha_scale <- sd(
  df$alpha_empirical,
  na.rm = TRUE
)

df <- df %>%
  mutate(
    alpha_z =
      (alpha_empirical - alpha_center) / alpha_scale
  )

# 3. Bayesian model: alpha across ADHD presentations ----

m_alpha_presentation <- brm(
  alpha_z ~ presentation3,
  
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

summary(m_alpha_presentation)

# 4. Posterior estimates for each presentation group ----

newdata_alpha_presentation <- tibble(
  presentation3 = factor(
    c(
      "Without ADHD",
      "Inattentive",
      "Combined/HI"
    ),
    levels = levels(df$presentation3)
  )
)

alpha_presentation_epred <- posterior_epred(
  m_alpha_presentation,
  newdata = newdata_alpha_presentation
)

# 5. Posterior summary by presentation group ----

alpha_presentation_summary <- tibble(
  group = c(
    "Without ADHD",
    "Inattentive",
    "Combined/HI"
  ),
  
  posterior_median = apply(
    alpha_presentation_epred, 2, median
  ),
  
  lower_90 = apply(
    alpha_presentation_epred, 2, quantile, probs = 0.05
  ),
  
  upper_90 = apply(
    alpha_presentation_epred, 2, quantile, probs = 0.95
  )
)

alpha_presentation_summary

# 6. Pairwise posterior contrasts ----

alpha_diff_inatt_vs_without <-
  alpha_presentation_epred[, 2] -
  alpha_presentation_epred[, 1]

alpha_diff_combined_vs_without <-
  alpha_presentation_epred[, 3] -
  alpha_presentation_epred[, 1]

alpha_diff_combined_vs_inatt <-
  alpha_presentation_epred[, 3] -
  alpha_presentation_epred[, 2]


alpha_presentation_contrasts <- tibble(
  contrast = c(
    "Inattentive - Without ADHD",
    "Combined/HI - Without ADHD",
    "Combined/HI - Inattentive"
  ),
  
  median = c(
    median(alpha_diff_inatt_vs_without),
    median(alpha_diff_combined_vs_without),
    median(alpha_diff_combined_vs_inatt)
  ),
  
  lower_90 = c(
    quantile(alpha_diff_inatt_vs_without, 0.05),
    quantile(alpha_diff_combined_vs_without, 0.05),
    quantile(alpha_diff_combined_vs_inatt, 0.05)
  ),
  
  upper_90 = c(
    quantile(alpha_diff_inatt_vs_without, 0.95),
    quantile(alpha_diff_combined_vs_without, 0.95),
    quantile(alpha_diff_combined_vs_inatt, 0.95)
  ),
  
  pd = c(
    max(
      mean(alpha_diff_inatt_vs_without > 0),
      mean(alpha_diff_inatt_vs_without < 0)
    ),
    max(
      mean(alpha_diff_combined_vs_without > 0),
      mean(alpha_diff_combined_vs_without < 0)
    ),
    max(
      mean(alpha_diff_combined_vs_inatt > 0),
      mean(alpha_diff_combined_vs_inatt < 0)
    )
  ) * 100
)

alpha_presentation_contrasts

# 7. Posterior difference: Combined/HI - Inattentive ----

alpha_subtype_diff_summary <- tibble(
  median = median(alpha_diff_combined_vs_inatt),
  lower_90 = quantile(alpha_diff_combined_vs_inatt, 0.05),
  upper_90 = quantile(alpha_diff_combined_vs_inatt, 0.95),
  pd = max(
    mean(alpha_diff_combined_vs_inatt > 0),
    mean(alpha_diff_combined_vs_inatt < 0)
  ) * 100
)

alpha_subtype_diff_summary

# 7.1 Plot: Combined/HI - Inattentive posterior difference ----

p_alpha_subtype_diff <- ggplot(
  tibble(diff = alpha_diff_combined_vs_inatt),
  aes(x = diff)
) +
  
  geom_density(
    fill = "#7B61A8",
    color = "#5F478C",
    alpha = 0.55,
    linewidth = 1
  ) +
  
  # Zero = no difference between ADHD presentations
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  # Posterior median
  annotate(
    "point",
    x = median(alpha_diff_combined_vs_inatt),
    y = 0,
    color = "black",
    size = 3
  ) +
  
  labs(
    x = "Estimated Combined/HI - Inattentive difference in α (SD units)",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    aspect.ratio = 0.45
  )

p_alpha_subtype_diff


# 7.2 Save subtype difference plot ----

ggsave(
  filename = "Figures/alpha/alpha_combinedHI_vs_inattentive_difference.png",
  plot = p_alpha_subtype_diff,
  width = 7,
  height = 3.5,
  dpi = 300,
  bg = "white"
)


# 8. Plot: posterior alpha distributions by ADHD presentation ----

library(tidyr)

alpha_presentation_long <- as_tibble(alpha_presentation_epred) %>%
  setNames(c("Without ADHD", "Inattentive", "Combined/HI")) %>%
  pivot_longer(
    cols = everything(),
    names_to = "group",
    values_to = "alpha_z"
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c("Without ADHD", "Inattentive", "Combined/HI")
    )
  )

alpha_presentation_medians <- alpha_presentation_long %>%
  group_by(group) %>%
  summarise(
    median = median(alpha_z),
    .groups = "drop"
  )

p_alpha_presentation <- ggplot(
  alpha_presentation_long,
  aes(x = alpha_z, fill = group, color = group)
) +
  geom_density(
    alpha = 0.35,
    linewidth = 1
  ) +
  
  geom_point(
    data = alpha_presentation_medians,
    aes(x = median, y = 0),
    inherit.aes = FALSE,
    color = "black",
    size = 3
  ) +
  
  scale_fill_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Without ADHD" = "#CC79A7",
      "Inattentive" = "#0072B2",
      "Combined/HI" = "#009E73"
    )
  ) +
  
  labs(
    x = "Estimated standardized alpha",
    y = NULL,
    fill = NULL,
    color = NULL
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

p_alpha_presentation

ggsave(
  filename = "Figures/alpha/alpha_posterior_by_ADHD_presentation.png",
  plot = p_alpha_presentation,
  width = 9,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

# 9. Posterior contrast: Combined/HI - Without ADHD ----

alpha_combined_vs_without_summary <- describe_posterior(
  alpha_diff_combined_vs_without,
  centrality = "median",
  ci = 0.90,
  test = "pd"
)

alpha_combined_vs_without_summary

# 9.1 Plot: Combined/HI - Without ADHD posterior difference ----

p_alpha_combined_vs_without <- ggplot(
  tibble(diff = alpha_diff_combined_vs_without),
  aes(x = diff)
) +
  
  geom_density(
    fill = "#009E73",
    color = "#007A5E",
    alpha = 0.55,
    linewidth = 1
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.8
  ) +
  
  annotate(
    "point",
    x = median(alpha_diff_combined_vs_without),
    y = 0,
    color = "black",
    size = 3
  ) +
  
  labs(
    x = "Estimated Combined/HI - Without ADHD difference in α (SD units)",
    y = NULL
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    
    aspect.ratio = 0.45
  )

p_alpha_combined_vs_without

# 9.2 save plot

ggsave(
  filename = "Figures/alpha/alpha_combinedHI_vs_withoutADHD_difference.png",
  plot = p_alpha_combined_vs_without,
  width = 7,
  height = 3.5,
  dpi = 300,
  bg = "white"
)