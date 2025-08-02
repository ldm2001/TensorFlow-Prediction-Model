<?php

session_start();

$ses_id    = (isset($_SESSION['ses_id'   ]) && $_SESSION['ses_id'   ] != '' ) ? $_SESSION['ses_id'   ] : '';
$ses_level = (isset($_SESSION['ses_level']) && $_SESSION['ses_level'] != '' ) ? $_SESSION['ses_level'] : '';