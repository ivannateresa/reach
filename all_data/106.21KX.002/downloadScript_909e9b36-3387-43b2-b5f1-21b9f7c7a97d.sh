#!/bin/sh

usage () {
    cat <<__EOF__
usage: $(basename "$0") [-hlp] [-u user] [-X args] [-d args]
  -h        print this help text
  -l        print list of files to download
  -p        prompt for password
  -u user   download as a different user
  -X args   extra arguments to pass to xargs
  -d args   extra arguments to pass to the download program

__EOF__
}

hostname=dataportal.eso.org
username=anonymous
anonymous=
xargsopts=
prompt=
list=
while getopts hlpu:xX:d: option
do
    case $option in
	h) usage; exit ;;
	l) list=yes ;;
	p) prompt=yes ;;
	u) prompt=yes; username="$OPTARG" ;;
	X) xargsopts="$OPTARG" ;;
	d) download_opts="$OPTARG";;
	?) usage; exit 2 ;;
    esac
done

if [ "$username" = "anonymous" ]; then
    anonymous=yes
fi

if [ -z "$xargsopts" ]; then
    #no xargs option specified, we ensure that only one url
    #after the other will be used
    xargsopts='-L 1'
fi

netrc=$HOME/.netrc
if [ -z "$anonymous" ] && [ -z "$prompt" ]; then
    # take password (and user) from netrc if no -p option
    if [ -f "$netrc" ] && [ -r "$netrc" ]; then
	grep -ir "$hostname" "$netrc" > /dev/null
	if [ $? -ne 0 ]; then
            #no entry for $hostname, user is prompted for password
            echo "A .netrc is available but there is no entry for $hostname, add an entry as follows if you want to use it:"
            echo "machine $hostname login anonymous password _yourpassword_"
            prompt="yes"
	fi
    else
	prompt="yes"
    fi
fi

if [ -n "$prompt" ] && [ -z "$list" ]; then
    trap 'stty echo 2>/dev/null; echo "Cancelled."; exit 1' INT HUP TERM
    stty -echo 2>/dev/null
    printf 'Password: '
    read password
    echo ''
    stty echo 2>/dev/null
    escaped_password=${password//\%/\%25}
    auth_check=$(wget -O - --post-data "username=$username&password=$escaped_password" --server-response --no-check-certificate "https://www.eso.org/sso/oidc/accessToken?grant_type=password&client_id=clientid" 2>&1 | awk '/^  HTTP/{print $2}')
    if [ ! "$auth_check" -eq 200 ]
    then
        echo 'Invalid password!'
        exit 1
    fi
fi

# use a tempfile to which only user has access 
tempfile=$(mktemp /tmp/dl.XXXXXXXX 2>/dev/null)
test "$tempfile" -a -f "$tempfile" || {
    tempfile=/tmp/dl.$$
    ( umask 077 && : >$tempfile )
}
trap 'rm -f $tempfile' EXIT INT HUP TERM

echo "auth_no_challenge=on" > "$tempfile"
# older OSs do not seem to include the required CA certificates for ESO
echo "check_certificate=off" >> "$tempfile"
echo "content_disposition=on" >> "$tempfile"
if [ -z "$anonymous" ] && [ -n "$prompt" ]; then
    echo "http_user=$username" >> "$tempfile"
    echo "http_password=$password" >> "$tempfile"
fi
WGETRC=$tempfile; export WGETRC

unset password

if [ -n "$list" ]; then
    cat
else
    xargs "$xargsopts" wget "$download_opts"
fi <<'__EOF__'
https://archive.eso.org/downloadportalapi/readme/909e9b36-3387-43b2-b5f1-21b9f7c7a97d
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:17:20.513_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:22:33.815_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:16:39.645_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:14:24.511_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:41:52.329_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:15:02.608_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:02:56.632_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:23:11.860_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:24:28.714_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:40:36.114_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:01:04.761_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:34:15.521_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:13:46.392_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:33:36.912_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:39:58.034_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:15:40.798_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:16:19.154_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:15:40.798_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:33:36.912_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:17:41.283_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:14:24.511_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:21:55.625_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:39:19.970_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:01:41.727_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:23:50.035_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:31:04.311_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:13:46.392_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:00:27.722_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:41:13.922_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:32:58.720_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:21:17.725_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:31:42.558_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:13:08.379_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T01:59:50.624_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:38:42.037_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:31:42.558_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:17:00.064_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:15:02.608_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:32:20.692_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:32:20.692_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:02:18.933_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:32:58.720_raw2raw.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:31:04.311_raw2master.xml
https://archive.eso.org/downloadportalapi/calibrationxml/909e9b36-3387-43b2-b5f1-21b9f7c7a97d/PIONI.2021-02-26T02:13:08.379_raw2raw.xml
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:00:27.722.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:27:02.303
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:01:04.761
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:19.380
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:51.896
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:13:08.379.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:15:02.608.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:31:04.311.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:39:19.970
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:48.470
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:32:20.692.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:39.673
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:41:52.329
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:15:40.798.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:25:53.276
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:24:28.714.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:27:21.366
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:01:41.727
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:15:40.798
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:01:04.761.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:40:36.114.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:11.923
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:33:36.912.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:58.740
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:53.600
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T01:59:50.624
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:00:27.722
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:23:11.860.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:39:58.034
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:16:39.645
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:21:17.725
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:43.346
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:21:55.625.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:34.446
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:22:33.815
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:41:13.922
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:31:42.558.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:23:50.035.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:27.020
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:57.020
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:34:15.521.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:13.903
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:17.606
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:32:58.720
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:34:15.521
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:39:19.970.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:14:24.511.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:10.063
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T01:59:50.624.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:41.283
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:40:36.114
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:21:17.725.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:02:56.632.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:14:24.511
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:23:50.035
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:27:35.693
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:20.513
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:24:28.714
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:28.820
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:22:33.815.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:41.616
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:00.064
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:16:39.645.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:13:08.379
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2023-07-25T09:40:28.670
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:31:04.311
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:41.283.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:15:02.608
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T09:16:58.944
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:23:11.860
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:39:58.034.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:46.726
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:32.666
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:01:41.727.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:13:46.392.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:21.413
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:21:55.625
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:36.246
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:41:52.329.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:38:42.037
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:38:42.037.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:16:19.154.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:32:20.692
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:41:13.922.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:32:58.720.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:33:36.912
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:27:41.986
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:26:24.956
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:02:18.933.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:13:46.392
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:31:42.558
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:20.513.NL
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:27:16.116
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:02:56.632
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:16:19.154
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:25:55.176
https://dataportal.eso.org/dataPortal/file/M.PIONIER.2021-03-02T14:25:57.443
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:17:00.064.NL
https://dataportal.eso.org/dataPortal/file/PIONI.2021-02-26T02:02:18.933
https://dataportal.eso.org/dataPortal/file/ADP.2021-12-15T20:32:44.443
https://dataportal.eso.org/dataPortal/file/ADP.2021-12-15T20:32:44.445
https://dataportal.eso.org/dataPortal/file/ADP.2021-12-15T20:32:44.444
https://dataportal.eso.org/dataPortal/file/ADP.2021-12-15T20:32:44.446
__EOF__
