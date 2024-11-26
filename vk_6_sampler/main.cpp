#include "application/application.h"
#include "vkInit.hpp"

int main() {
    App->init(1920, 1080);

    while (App->update()) {

        vkInit::Base().drawFrame();

    }

    App->end();

    return 0;
}