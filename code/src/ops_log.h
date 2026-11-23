#pragma once
#include "souris/session_manager.h"
#include <fstream>

namespace souris {
/** Request-only JSONL, with daily rotation and no prompt/token contents. */
class OpsLog {
public:
    explicit OpsLog(const Config &);
    void request_end(const SessionManager &, const CompletionResult &,
                     const std::string & request_id, const std::string & error = "");
    void error(const std::string &, const std::string & request_id = "", const std::string & session_id = "");
private:
    std::string path_, day_;
    bool errors_only_ = false;
    std::string model_;
    int mmq_group_;
    std::ofstream file_;
    void open_file();
    void write(const std::string &);
};
}
