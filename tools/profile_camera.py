from onvif import ONVIFCamera
import re
import time
import platform    # For getting the operating system name
import subprocess  # For executing a shell command
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ping(host):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """
    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c'
    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', host]
    return subprocess.call(command) == 0

# def run_wsdiscovery(ip):
#     reg = re.compile("^http:\/\/(\d*\.\d*\.\d*\.\d*).*\/onvif")
#     wsd = WSDiscovery()
#     ip_range = [ip]
#     wsd.start()
#     ws_devices = wsd.searchServiceInRange(ip_range) # take 3 seconds
#     ip_addresses = []
#     for ws_device_range in ws_devices:
#         for ws_device in ws_device_range:
#             for http_address in ws_device.getXAddrs():
#                 m = reg.match(http_address)
#                 if m is None:
#                     continue
#                 else:
#                     ip_address = http_address[m.start(1):m.end(1)]
#                     ip_addresses.append(ip_address)
#     wsd.stop()
#     return ip_addresses

def profiling_camera(ip, user: str, password: str):
    ping_status = ping(ip)
    if not ping_status:
        return "Unreachable"
    mycam = ONVIFCamera(ip, 80, user, password, 'wsdl')
    media_service = mycam.create_media_service()
    profiles = media_service.GetProfiles()
    rtsp_list = []
    for profile in profiles:
        token = profile.token
        obj = media_service.create_type('GetStreamUri')
        obj.ProfileToken = token
        obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
        stream_uri = media_service.GetStreamUri(obj)
        if stream_uri.Uri not in rtsp_list:
            rtsp_list.append(stream_uri.Uri)
    configurations_list = media_service.GetVideoEncoderConfigurations()
    stream_config_list = []
    for config in configurations_list:
        stream_data = []
        stream_data.append('encoding/' + config.Encoding)
        stream_data.append('width/' + str(config.Resolution.Width))
        stream_data.append('height/' + str(config.Resolution.Height))
        stream_data.append('framerate_limit/' + str(config.RateControl.FrameRateLimit))
        stream_data.append('bitrate_limit/' + str(config.RateControl.BitrateLimit))
        stream_data = "/".join(stream_data)
        stream_config_list.append(stream_data)
    return {"rtsp_uri": ",".join(rtsp_list), "stream": ",".join(stream_config_list)}
if __name__ == "__main__":
    # print(run_wsdiscovery())
    res = profiling_camera('172.21.104.100', 'admin', '123456a@')
    print(res)
    # print(run_wsdiscovery('172.21.104.100'))
