#pragma once
#include "souris/session_manager.h"

namespace souris {
class OpsLog;
class Runtime;
/** Raw completion HTTP loop. One worker owns the session through delivery. */
int serve(Runtime &, OpsLog &);
}
