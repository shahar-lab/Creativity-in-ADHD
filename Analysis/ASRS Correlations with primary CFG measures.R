# ============================================================
# ASRS CORRELATIONS WITH PRIMARY CFG MEASURES
# All participants and ADHD only
# ============================================================


# 1. Packages ----

library(tidyverse)
library(gt)
library(webshot2)

# 2. Load data ----

df <- read_csv(
  "Data/df_creativity_ADHD.csv",
  show_col_types = FALSE
)


# 3. Prepare variables for correlation analysis ----

correlation_df <- df %>%
  transmute(
    
    # Diagnostic group
    diva_group = diva_group,
    
    # ASRS symptom scores
    ASRS_Total = asrs,
    ASRS_Inattention = asrs_ia,
    ASRS_HI = asrs_hi,
    
    # Primary CFG measures
    Originality = `Gallery Orig`,
    Fluency = `#galleries`,
    g = g_empirical,
    Alpha = alpha_empirical,
    Efficiency_Exploration = `exp efficiency`,
    Efficiency_Exploitation = `scav efficiency`
  )

# 4. Initial checks ----

nrow(correlation_df)

table(
  correlation_df$diva_group,
  useNA = "ifany"
)

correlation_df %>%
  summarise(
    across(
      everything(),
      ~ sum(is.na(.x))
    )
  )

# 4.1 Available ASRS data ----

correlation_df %>%
  summarise(
    n_total = n(),
    n_ASRS_Total = sum(!is.na(ASRS_Total)),
    n_ASRS_Inattention = sum(!is.na(ASRS_Inattention)),
    n_ASRS_HI = sum(!is.na(ASRS_HI))
  )

correlation_df %>%
  group_by(diva_group) %>%
  summarise(
    n = n(),
    n_ASRS_Total = sum(!is.na(ASRS_Total)),
    n_ASRS_Inattention = sum(!is.na(ASRS_Inattention)),
    n_ASRS_HI = sum(!is.na(ASRS_HI)),
    .groups = "drop"
  )

# 5. Function for Spearman correlations ----

run_spearman_correlations <- function(data) {
  
  asrs_vars <- c(
    "ASRS_Total",
    "ASRS_Inattention",
    "ASRS_HI"
  )
  
  cfg_vars <- c(
    "Originality",
    "Fluency",
    "g",
    "Alpha",
    "Efficiency_Exploration",
    "Efficiency_Exploitation"
  )
  
  crossing(
    CFG_measure = cfg_vars,
    ASRS_measure = asrs_vars
  ) %>%
    
    mutate(
      
      result = map2(
        CFG_measure,
        ASRS_measure,
        
        ~ {
          
          temp <- data %>%
            select(
              all_of(.x),
              all_of(.y)
            ) %>%
            drop_na()
          
          test <- cor.test(
            temp[[.x]],
            temp[[.y]],
            method = "spearman",
            exact = FALSE
          )
          
          tibble(
            rho = unname(test$estimate),
            p_value = test$p.value,
            n = nrow(temp)
          )
        }
      )
    ) %>%
    
    unnest(result) %>%
    
    mutate(
      p_holm = p.adjust(
        p_value,
        method = "holm"
      )
    )
}

# 6. Correlations - All participants ----

correlations_all <- run_spearman_correlations(
  correlation_df
)

correlations_all

# 7. Correlations - ADHD only ----

correlation_df_adhd <- correlation_df %>%
  filter(diva_group == "ADHD")

correlations_adhd <- run_spearman_correlations(
  correlation_df_adhd
)

correlations_adhd

# 8. Create and save correlation matrices ----

measure_order <- c(
  "Originality",
  "Fluency",
  "g",
  "Alpha",
  "Efficiency_Exploration",
  "Efficiency_Exploitation"
)


# 8.1 All participants matrix ----

correlation_table_all <- correlations_all %>%
  select(
    CFG_measure,
    ASRS_measure,
    rho
  ) %>%
  mutate(
    CFG_measure = factor(
      CFG_measure,
      levels = measure_order
    )
  ) %>%
  pivot_wider(
    names_from = ASRS_measure,
    values_from = rho
  ) %>%
  arrange(CFG_measure) %>%
  select(
    CFG_measure,
    ASRS_Total,
    ASRS_Inattention,
    ASRS_HI
  ) %>%
  mutate(
    CFG_measure = as.character(CFG_measure),
    CFG_measure = recode(
      CFG_measure,
      "Originality" = "Originality",
      "Fluency" = "Fluency",
      "g" = "g",
      "Alpha" = "α",
      "Efficiency_Exploration" = "Efficiency – Exploration",
      "Efficiency_Exploitation" = "Efficiency – Exploitation"
    ),
    across(
      where(is.numeric),
      ~ round(.x, 2)
    )
  ) %>%
  rename(
    `CFG measure` = CFG_measure,
    `ASRS Total` = ASRS_Total,
    `Inattention` = ASRS_Inattention,
    `HI` = ASRS_HI
  )


table_all <- correlation_table_all %>%
  gt() %>%
  tab_header(
    title = "Associations between ASRS symptoms and CFG measures",
    subtitle = "All participants (N = 155) | Spearman's ρ"
  ) %>%
  tab_spanner(
    label = "ASRS",
    columns = c(
      `ASRS Total`,
      `Inattention`,
      `HI`
    )
  )

table_all

# 8.2 ADHD-only matrix ----

correlation_table_adhd <- correlations_adhd %>%
  select(
    CFG_measure,
    ASRS_measure,
    rho
  ) %>%
  mutate(
    CFG_measure = factor(
      CFG_measure,
      levels = measure_order
    )
  ) %>%
  pivot_wider(
    names_from = ASRS_measure,
    values_from = rho
  ) %>%
  arrange(CFG_measure) %>%
  select(
    CFG_measure,
    ASRS_Total,
    ASRS_Inattention,
    ASRS_HI
  ) %>%
  mutate(
    CFG_measure = as.character(CFG_measure),
    CFG_measure = recode(
      CFG_measure,
      "Originality" = "Originality",
      "Fluency" = "Fluency",
      "g" = "g",
      "Alpha" = "α",
      "Efficiency_Exploration" = "Efficiency – Exploration",
      "Efficiency_Exploitation" = "Efficiency – Exploitation"
    ),
    across(
      where(is.numeric),
      ~ round(.x, 2)
    )
  ) %>%
  rename(
    `CFG measure` = CFG_measure,
    `ASRS Total` = ASRS_Total,
    `Inattention` = ASRS_Inattention,
    `HI` = ASRS_HI
  )


table_adhd <- correlation_table_adhd %>%
  gt() %>%
  tab_header(
    title = "Associations between ASRS symptoms and CFG measures",
    subtitle = "ADHD group only (N = 79) | Spearman's ρ"
  ) %>%
  tab_spanner(
    label = "ASRS",
    columns = c(
      `ASRS Total`,
      `Inattention`,
      `HI`
    )
  )

table_adhd

# 8.3 Save correlation tables ----

output_dir <- file.path(
  getwd(),
  "Figures",
  "ASRS_correlations"
)

dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

gt::gtsave(
  data = table_all,
  filename = "ASRS_correlations_all.png",
  path = output_dir
)

gt::gtsave(
  data = table_adhd,
  filename = "ASRS_correlations_ADHD_only.png",
  path = output_dir
)

# 9. Optional inferential checks ----

correlations_adhd %>%
  arrange(ASRS_measure, desc(abs(rho))) %>%
  mutate(
    rho = round(rho, 3),
    p_value = round(p_value, 4),
    p_holm = round(p_holm, 4)
  )