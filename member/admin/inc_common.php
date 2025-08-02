<?php

session_start();

$ses_id    = (isset($_SESSION['ses_id'   ]) &&  $_SESSION['ses_id'] != ''   ) ? $_SESSION['ses_id'   ] : '';
$ses_level = (isset($_SESSION['ses_level']) &&  $_SESSION['ses_level'] != '') ? $_SESSION['ses_level'] : ''; 

if($ses_id == '' || $ses_level != 10) {
  die("
  <script>
    alert('현재 계정은 관리자 계정이 아닙니다.');
    self.location.href='../';
  </script>");
}