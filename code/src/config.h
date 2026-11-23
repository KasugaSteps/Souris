#pragma once
#include "souris/types.h"

namespace souris {
/** Parse the startup TOML once, rejecting unknown keys and invalid budgets. */
Config read_config(const std::string & path);
}
