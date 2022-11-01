#include "faceApp.h"

int main(int argc, char *argv[])
{
	QDTLog::init("logs/");

	FaceApp app(argc, argv);
	app.init();
	app.run();

	return 0;
}
