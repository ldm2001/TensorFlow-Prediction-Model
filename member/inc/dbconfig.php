<?php

$servername = '127.0.0.1';
$dbuser = 'root';
$dbpassword = '';
$dbname = 'memsite';

try {
  $db = new PDO("mysql:host=localhost;port=3300;dbname={$dbname}", $dbuser, $dbpassword);

  $db->setAttribute(PDO::ATTR_EMULATE_PREPARES, false);
  $db->setAttribute(PDO::MYSQL_ATTR_USE_BUFFERED_QUERY, true);
  $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

} catch (PDOException $e) {
  echo $e->getMessage();
}

define('DOCUMENT_ROOT', $_SERVER['DOCUMENT_ROOT'] .'/project/member');
define('ADMIN_DIR'    , DOCUMENT_ROOT .'/admin'  );
define('DATA_DIR'     , DOCUMENT_ROOT .'/data'   );
define('PROFILE_DIR'  , DATA_DIR      .'/profile');
define('BOARD_DIR'    , DATA_DIR      .'/board'  ); // 파일이 저장될 절대 경로 
define('BOARD_WEB_DIR', 'data/board'             ); // 웹에서 확인하는 경로
define('POPUP_DIR'    , DATA_DIR      .'/popup'  ); // 파일이 저장될 절대 경로 
define('POPUP_WEB_DIR', 'data/popup'             ); // 웹에서 확인하는 경로