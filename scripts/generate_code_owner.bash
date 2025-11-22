# bash syntax function for current directory git repository
owners(){
  for f in $(git ls-files | sort -u); do
    # directory path
    echo -n "$f "
    # authors if loc distribution >= 30%
    git fame -snwMC --incl "$f" | tr '/' '|' \
      | awk -F '|' '(NR>6 && $6>=30) { sub(/^ +/, "", $2); print "@"$2 }' \
      | xargs echo
  done
}

# print to screen and file
owners | tee .github/CODEOWNERS

# same but with `tqdm` progress for large repos
owners \
  | tqdm --total $(git ls-files | wc -l) \
    --unit file --desc "Generating CODEOWNERS" \
  > .github/CODEOWNERS

# Replace all the names with GitHub usernames
python scripts/name_replace.py .github/name_mappings.csv .github/CODEOWNERS
