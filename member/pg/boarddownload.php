<?php
include '../inc/common.php';

if($ses_id == '') {
  die('<script>alert("로그인한 회원만 다운로드가 가능합니다.");
  self.location.href="../login.php"
  </script>');
}

$idx = (isset($_GET['idx']) && $_GET['idx'] != '' && is_numeric($_GET['idx'])) ? $_GET['idx'] : '';
$th  = (isset($_GET['th' ]) && $_GET['th' ] != '' && is_numeric($_GET['th' ])) ? $_GET['th' ] : '';

if($idx == '') {
  die('<script>alert("게시물 번호가 빠졌습니다.");</script>');
}

if($th == '') {
  die('<script>alert("몇 번째 파일인지 알수가 없습니다.");</script>');
}

include '../inc/dbconfig.php';
include '../inc/board.php'; 

$board = new Board($db);

$fileinfo = $board->getAttachFile($idx, $th);
list($file_source, $file_name, $totalfile) = explode('|', $fileinfo);

// 다운로드 수 구하기
$downhit = $board->getDownhit($idx);

if($downhit == '') {
  $tmp_arr = [];
  for($i = 0; $i < $totalfile; $i++) {
    if($th == $i) {
      $tmp_arr[] = 1;
    } else {
      $tmp_arr[] = 0;
    }
  }   
} else {
  // $downhit = '4?0';
  $tmp_arr = explode('?', $downhit);
  $tmp_arr[$th] = $tmp_arr[$th] + 1;
}

$downhit_str = implode('?', $tmp_arr);

$str = $idx .'?'. $th;
if(isset($_SESSION['lastdownhit']) && $_SESSION['lastdownhit'] != '') {
  
  if($str != $_SESSION['lastdownhit']) {
    $board->increaseDownhit($idx, $downhit_str); 
    $_SESSION['lastdownhit'] = $str;
  }
} else {
  $board->increaseDownhit($idx, $downhit_str); 
  $_SESSION['lastdownhit'] = $str;
}

if($file_source == '' || $file_name == '') {
  die('<script>alert("파일 정보를 제대로 가져오지 못했습니다.");</script>');
}

$down = BOARD_DIR .'/'. $file_source;

if(!file_exists($down)) {
  die('<script>alert("존재하지 않는파일입니다.");</script>');
}

$filesize = filesize($down);

header("Content-Type:application/octet-stream");
header("Content-Disposition:attachment;filename=$file_name"); 
header("Content-Transfer-Encoding:binary");
header("Content-Length:$filesize");
header("Cache-Control:cache,must-revalidate");
header("Pragma:no-cache");

$fp = fopen($down, "r");
while (!feof($fp)) {
  $buf = fread($fp, 8096);
  print($buf);
  flush();
}

fclose($fp);
?>