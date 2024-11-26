#include "application/application.h"
#include "vkInit.hpp"

int main() {
    App->init(800, 600);

    while (App->update()) {

        vkInit::Base().drawFrame();

    }

    App->end();

    return 0;
}