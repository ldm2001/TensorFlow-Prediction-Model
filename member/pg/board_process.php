<?php 

if(isset($_SERVER['CONTENT_LENGTH']) && $_SERVER['CONTENT_LENGTH'] >  (int) ini_get('post_max_size') * 1024 * 1024) {

  $arr = ['result' => 'post_size_exceed'];
  die(json_encode($arr));
}

include '../inc/common.php';
include '../inc/dbconfig.php';
include '../inc/board.php';  
include '../inc/member.php'; 

$mode    = (isset($_POST['mode'   ]) && $_POST['mode'   ] != '') ? $_POST['mode'   ] : '';
$bcode   = (isset($_POST['bcode'  ]) && $_POST['bcode'  ] != '') ? $_POST['bcode'  ] : '';
$subject = (isset($_POST['subject']) && $_POST['subject'] != '') ? $_POST['subject'] : '';
$content = (isset($_POST['content']) && $_POST['content'] != '') ? $_POST['content'] : '';
$idx     = (isset($_POST['idx'    ]) && $_POST['idx'    ] != '' && is_numeric($_POST['idx'])) ? $_POST['idx'] : '';
$th      = (isset($_POST['th'     ]) && $_POST['th'     ] != '' && is_numeric($_POST['th' ])) ? $_POST['th' ] : '';

if($mode == '') {
  $arr = ["result" => "empty_mode"];
  $json_str = json_encode($arr);  // 배열 -> json 문자열
  die($json_str);
}

if($bcode == '') {
  $arr = ["result" => "empty_bcode"];
  die(json_encode($arr));
}

$board = new Board($db); 
$member = new Member($db);

if($mode == 'input') {

  // 이미지 변환하여 저장하기
  preg_match_all("/<img[^>]*src=[\"']?([^>\"']+)[\"']?[^>]*>/i", $content, $matches);
  $img_array = [];
  foreach($matches[1] AS $key => $row) {
    if(substr($row, 0, 5) != 'data:') {
      continue;
    }

    // data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAI
    list($type, $data) = explode(';', $row);
    list(,$data) = explode(',', $data);
    $data = base64_decode($data);
    list(,$ext) = explode('/', $type);
    $ext = ($ext == 'jpeg') ? 'jpg' : $ext;

    $filename = date('YmdHis') .'_'. $key .'.'. $ext;

    file_put_contents(BOARD_DIR."/". $filename, $data);

    $content = str_replace($row, BOARD_WEB_DIR."/". $filename, $content);
    $img_array[] = BOARD_WEB_DIR."/". $filename;
  }
  
  if($subject == '') {
    die(json_encode(["result" => "empty_subject"]));
  }

  if($content == '' || $content == '<p><br></p>') {
    die(json_encode(["result" => "empty_content"]));
  }

  // 파일첨부
  $file_list_str = '';
  $file_cnt = 3;
  if(isset($_FILES['files'])) {
    $file_list_str = $board->file_attach($_FILES['files'], $file_cnt);
  }
  
  $memArr = $member->getInfo($ses_id);
  $name = $memArr['name'];
  
  $arr = [
    'bcode' => $bcode,
    'id' => $ses_id,
    'name' => $name,
    'subject' => $subject,
    'content' => $content,
    'files' => $file_list_str,
    'ip' => $_SERVER['REMOTE_ADDR']
  ];

  $board->input($arr);

  die(json_encode(["result" => "success"]));
}
else if($mode == 'each_file_del') {

  if($idx == '') {
    $arr = [ "result" => "empty_idx"];
    die(json_encode($arr)); 
  }
  if($th == '') {
    $arr = [ "result" => "empty_th"];
    die(json_encode($arr)); 
  }

  $file = $board->getAttachFile($idx, $th);

  $each_files = explode('|', $file);

  //  BOARD_DIR . '/'. $each_files[0]
  if(file_exists(BOARD_DIR . '/'. $each_files[0])) {
    unlink(BOARD_DIR . '/'. $each_files[0]);
  }

  $row = $board->view($idx);
  // $row['files']
  $files = explode('?', $row['files']);
  $tmp_arr = []; 
  foreach($files AS $key => $val) {

    if ($key == $th) {
      continue;
    }

    $tmp_arr[] = $val;
  }

  $files = implode('?', $tmp_arr); 

  $tmp_arr = [];
  $downs = explode('?', $row['downhit']);
  foreach($downs AS $key => $val) {

    if ($key == $th) {
      continue;
    }

    $tmp_arr[] = $val;
  }
  
  $downs = implode('?', $tmp_arr); // 새로 조합된 다운로드수 문자열

  $board->updateFileList($idx, $files, $downs); 

  $arr = [ "result" => "success"];
  die(json_encode($arr)); 

}
else if($mode == 'file_attach') {
  
  // 수정에서 개별파일 첨부하기
  $file_list_str = '';
  if(isset($_FILES['files'])) {
    $file_cnt = 1;
    $file_list_str = $board->file_attach($_FILES['files'], $file_cnt);
  } else {
    $arr = [ "result" => "empty_files"];
    die(json_encode($arr)); 
  }

  $row = $board->view($idx);

  if($row['files'] != '') {
    $files = $row['files'] .'?'. $file_list_str;
  }else {
    $files = $file_list_str;
  }

  if($row['downhit'] != '') {
    $downs = $row['downhit'] .'?0';
  } else {
    $downs = '';
  }

  $board->updateFileList($idx, $files, $downs); 

  $arr = [ "result" => "success"];
  die(json_encode($arr)); 

}
else if ($mode == 'edit') {

  $row = $board->view($idx);
  if($row['id'] != $ses_id) {
    die(json_encode(["result" => "permission_denied"]));
  }

  $old_image_arr = $board->extract_image($row['content']);
  
  // 이미지 변환하여 저장하기
  preg_match_all("/<img[^>]*src=[\"']?([^>\"']+)[\"']?[^>]*>/i", $content, $matches);
  $current_image_arr = [];
  foreach($matches[1] AS $key => $row) {
    if(substr($row, 0, 5) != 'data:') {
      $current_image_arr[] = $row;
      continue;
    }

    // data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAI
    list($type, $data) = explode(';', $row);
    list(,$data) = explode(',', $data);
    $data = base64_decode($data);
    list(,$ext) = explode('/', $type);
    $ext = ($ext == 'jpeg') ? 'jpg' : $ext;

    $filename = date('YmdHis') .'_'. $key .'.'. $ext;

    file_put_contents(BOARD_DIR."/". $filename, $data); // 파일업로드

    $content = str_replace($row, BOARD_WEB_DIR."/". $filename, $content); 
  }

  $diff_img_arr = array_diff($old_image_arr, $current_image_arr);
  foreach($diff_img_arr AS $value) {
    unlink("../".$value);
  }

  if($subject == '') {
    die(json_encode(["result" => "empty_subject"]));
  }

  if($content == '' || $content == '<p><br></p>') {
    die(json_encode(["result" => "empty_content"]));
  }

  $arr = [
    'idx' => $idx,
    'subject' => $subject,
    'content' => $content
  ];

  $board->edit($arr);

  die(json_encode(["result" => "success"]));

}
else if($mode == 'delete') {

  $row = $board->view($idx);
  
  // 본문이미지 삭제
  $img_arr = $board->extract_image($row['content']);
  foreach($img_arr AS $value) {  // dddd.png
    if(file_exists("../".$value)) {
      unlink("../".$value);
    }
  }

  // 첨부파일 삭제
  if($row['files'] != '') {
    $filelist = explode('?', $row['files']);
    foreach($filelist AS $value) {
      list($file_src, ) = explode('|', $value);
      unlink(BOARD_DIR .'/'. $file_src);
    }
  }

  $board->delete($idx);

  die(json_encode(["result" => "success"]));
}